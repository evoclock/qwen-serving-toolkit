import { randomUUID } from "node:crypto";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";

const TRACE_ENDPOINT =
	process.env.QWEN38FN_TRACE_URL ?? "http://127.0.0.1:8000/v1/reasoning-traces";
const TRACE_TOKEN = process.env.QWEN38FN_TRACE_TOKEN;
const MAX_TOOL_BYTES = 48_000;

type TraceAction = "list" | "get" | "probe";
type TraceFormat = "summary" | "json" | "text";

type TraceParams = {
	action: TraceAction;
	format?: TraceFormat;
	id?: string;
	session_id?: string;
	request_id?: string;
	limit?: number;
	prompt?: string;
	budget?: number;
	effort?: string;
	max_tokens?: number;
};

type TraceRecord = {
	id?: string;
	timestamp_ms?: number;
	task_id?: number;
	completion_id?: string;
	session_id?: string | null;
	request_id?: string | null;
	prompt_tokens?: number;
	completion_tokens?: number;
	reasoning_budget?: {
		enabled?: boolean;
		budget_tokens?: number;
		counted_tokens?: number;
		reasoning_tokens?: number;
		remaining_tokens?: number;
		termination?: string;
		end_sequence_emitted?: boolean;
		end_tag?: string | null;
	};
	reasoning_content?: string;
	content?: string;
	prompt?: string | null;
	response?: unknown;
};

type TraceList = { traces?: TraceRecord[]; count?: number };

type TraceEntry = {
	status: "saved" | "unavailable";
	endpoint: string;
	trace_id?: string;
	session_id: string;
	request_id: string;
	reasoning_budget?: TraceRecord["reasoning_budget"];
	reasoning_content?: string;
	content?: string;
	prompt?: string | null;
	error?: string;
};

function headers(): Record<string, string> {
	return TRACE_TOKEN
		? { Accept: "application/json", Authorization: `Bearer ${TRACE_TOKEN}` }
		: { Accept: "application/json" };
}

function completionsEndpoint(): URL {
	if (process.env.QWEN38FN_API_URL) {
		return new URL(`${process.env.QWEN38FN_API_URL.replace(/\/$/, "")}/v1/chat/completions`);
	}
	const url = new URL(TRACE_ENDPOINT);
	url.pathname = url.pathname.replace(/\/reasoning-traces$/, "/chat/completions");
	return url;
}

async function fetchTrace(
	query: Record<string, string | number | undefined>,
	signal?: AbortSignal,
): Promise<{ value: unknown; contentType: string }> {
	const url = new URL(TRACE_ENDPOINT);
	for (const [key, value] of Object.entries(query)) {
		if (value !== undefined && value !== "") url.searchParams.set(key, String(value));
	}

	const response = await fetch(url, { headers: headers(), signal });
	const body = await response.text();
	if (!response.ok) {
		throw new Error(`${response.status} ${response.statusText}: ${body.slice(0, 500)}`);
	}

	const contentType = response.headers.get("content-type") ?? "";
	if (contentType.includes("json")) {
		return { value: JSON.parse(body), contentType };
	}
	return { value: body, contentType };
}

async function runProbe(
	params: TraceParams,
	sessionId: string,
	signal?: AbortSignal,
): Promise<{ requestId: string; trace?: TraceRecord; response: unknown }> {
	if (!params.prompt) throw new Error("action=probe requires prompt");
	const budget = params.budget ?? 512;
	const requestId = `pi-probe-${randomUUID()}`;
	const url = completionsEndpoint();
	const body = {
		model: process.env.QWEN38FN_MODEL ?? "qwen3.8-flash-next",
		messages: [{ role: "user", content: params.prompt }],
		max_tokens: params.max_tokens ?? Math.min(32768, budget + 256),
		temperature: 0.0,
		top_p: 0.95,
		top_k: 20,
		reasoning_effort: params.effort ?? "xhigh",
		reasoning_budget_tokens: budget,
		reasoning_budget_message: `[PI_QWEN38FN_PROBE_${requestId}]`,
		trace_include_prompt: true,
		logprobs: true,
		top_logprobs: 5,
		stream: false,
	};
	const requestHeaders = {
		...headers(),
		"Content-Type": "application/json",
		"X-Session-ID": sessionId,
		"X-Request-ID": requestId,
	};
	const response = await fetch(url, {
		method: "POST",
		headers: requestHeaders,
		body: JSON.stringify(body),
		signal,
	});
	const responseText = await response.text();
	if (!response.ok) {
		throw new Error(`${response.status} ${response.statusText}: ${responseText.slice(0, 500)}`);
	}
	const value = JSON.parse(responseText);
	const trace = await findTrace(requestId, sessionId, signal);
	return { requestId, trace, response: value };
}

function maxReturnedTopLogprobs(trace: TraceRecord | undefined): number {
	const response = trace?.response;
	if (!response || typeof response !== "object" || Array.isArray(response)) return 0;
	const choices = (response as { choices?: unknown }).choices;
	if (!Array.isArray(choices)) return 0;
	let max = 0;
	for (const choice of choices) {
		if (!choice || typeof choice !== "object") continue;
		const logprobs = (choice as { logprobs?: unknown }).logprobs;
		if (!logprobs || typeof logprobs !== "object") continue;
		const content = (logprobs as { content?: unknown }).content;
		if (!Array.isArray(content)) continue;
		for (const token of content) {
			if (!token || typeof token !== "object") continue;
			const top = (token as { top_logprobs?: unknown }).top_logprobs;
			if (Array.isArray(top)) max = Math.max(max, top.length);
		}
	}
	return max;
}

function renderToolOutput(value: unknown): string {
	const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
	if (text.length <= MAX_TOOL_BYTES) return text;
	return `${text.slice(0, MAX_TOOL_BYTES)}\n\n[trace output truncated at ${MAX_TOOL_BYTES} bytes]`;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new Error("trace request cancelled"));
			return;
		}
		const timer = setTimeout(resolve, ms);
		const abort = () => {
			clearTimeout(timer);
			reject(new Error("trace request cancelled"));
		};
		signal?.addEventListener("abort", abort, { once: true });
	});
}

async function findTrace(requestId: string, sessionId: string, signal?: AbortSignal): Promise<TraceRecord | undefined> {
	for (let attempt = 0; attempt < 8; attempt++) {
		try {
			const result = await fetchTrace({ request_id: requestId, session_id: sessionId, limit: 5 }, signal);
			const traces = (result.value as TraceList)?.traces ?? [];
			const match = traces.find((trace) => trace.request_id === requestId);
			if (match) return match;
		} catch {
			// The final assistant event can arrive just before the trace file is
			// visible. Retry briefly; tracing must never break the model turn.
		}
		if (attempt < 7) await sleep(250, signal);
	}
	return undefined;
}

export default function (pi: ExtensionAPI) {
	const pendingRequestIds: string[] = [];

	pi.registerTool({
		name: "qwen38fn_trace",
		label: "Qwen38fn Trace",
		description:
			"Read Qwen3.8-Flash-Next reasoning traces from the DGX Spark over its read-only HTTP trace endpoint. Use for budget, token-count, termination, reasoning, answer, or optional logprob diagnostics. Never use shell commands or curl for this.",
		promptSnippet: "Query the DGX Spark Qwen38fn reasoning trace endpoint",
		promptGuidelines: [
			"Use qwen38fn_trace when the user asks for Qwen38fn reasoning budgets, termination, traces, or logprob diagnostics.",
			"Use qwen38fn_trace with action=list before action=get unless the user supplies an exact trace id.",
		],
		parameters: Type.Object({
			action: StringEnum(["list", "get", "probe"] as const),
			format: Type.Optional(StringEnum(["summary", "json", "text"] as const)),
			id: Type.Optional(Type.String({ description: "Trace JSON filename returned by action=list" })),
			session_id: Type.Optional(Type.String({ description: "Filter by Pi session UUID" })),
			request_id: Type.Optional(Type.String({ description: "Filter by provider request ID" })),
			limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 100 })),
			prompt: Type.Optional(Type.String({ description: "Prompt for a separate no-tools logprob diagnostic (action=probe)" })),
			budget: Type.Optional(Type.Integer({ minimum: 1, maximum: 16384 })),
			effort: Type.Optional(Type.String()),
			max_tokens: Type.Optional(Type.Integer({ minimum: 1, maximum: 32768 })),
		}),

		async execute(_toolCallId, params: TraceParams, signal, _onUpdate, ctx) {
			if (params.action === "get" && !params.id) {
				throw new Error("action=get requires id from a previous action=list result");
			}

			if (params.action === "probe") {
				const sessionId = ctx.sessionManager.getSessionId();
				const probe = await runProbe(params, sessionId, signal);
				const summary = {
					saved: Boolean(probe.trace),
					trace_id: probe.trace?.id,
					request_id: probe.requestId,
					session_id: sessionId,
					reasoning_budget: probe.trace?.reasoning_budget,
					prompt_tokens: probe.trace?.prompt_tokens,
					completion_tokens: probe.trace?.completion_tokens,
					max_returned_top_logprobs: maxReturnedTopLogprobs(probe.trace),
				};
				if (probe.trace) {
					pi.appendEntry("qwen38fn-reasoning-trace", {
						status: "saved",
						endpoint: TRACE_ENDPOINT,
						trace_id: probe.trace.id,
						session_id: sessionId,
						request_id: probe.requestId,
						reasoning_budget: probe.trace.reasoning_budget,
						reasoning_content: probe.trace.reasoning_content,
						content: probe.trace.content,
						prompt: probe.trace.prompt,
					});
				}
				return {
					content: [{ type: "text", text: renderToolOutput(summary) }],
					details: { endpoint: TRACE_ENDPOINT, action: "probe", ...summary },
				};
			}

			const result = await fetchTrace(
				{
					id: params.id,
					format: params.action === "get" && params.format === "text" ? "text" : undefined,
					session_id: params.action === "list" ? params.session_id : undefined,
					request_id: params.action === "list" ? params.request_id : undefined,
					limit: params.action === "list" ? params.limit ?? 10 : undefined,
				},
				signal,
			);
			const value = params.action === "get" && params.format === "summary"
				? summarizeTrace(result.value as TraceRecord)
				: result.value;

			return {
				content: [{ type: "text", text: renderToolOutput(value) }],
				details: {
					endpoint: TRACE_ENDPOINT,
					action: params.action,
					id: params.id,
				},
			};
		},
	});

	// Make every Qwen request addressable from both the Spark and Mac Pi
	// session stores. This is native Pi HTTP-header plumbing, not a shell hook.
	pi.on("before_provider_headers", (event, ctx) => {
		if (ctx.model?.provider !== "qwen38fn") return;
		const sessionId = ctx.sessionManager.getSessionId();
		const requestId = `pi-${randomUUID()}`;
		event.headers["X-Session-ID"] = sessionId;
		event.headers["X-Request-ID"] = requestId;
		pendingRequestIds.push(requestId);
	});

	// Copy the authoritative server-side diagnostic into the local Pi session
	// as a custom entry. Custom entries persist in JSONL but are not sent back
	// to the model, so they cannot inflate long-context prompts.
	pi.on("message_end", async (event, ctx) => {
		if (event.message.role !== "assistant") return;
		const requestId = pendingRequestIds.shift();
		if (!requestId) return;
		const sessionId = ctx.sessionManager.getSessionId();
		try {
			const trace = await findTrace(requestId, sessionId, ctx.signal);
			const entry: TraceEntry = trace
				? {
					  status: "saved",
					  endpoint: TRACE_ENDPOINT,
					  trace_id: trace.id,
					  session_id: sessionId,
					  request_id: requestId,
					  reasoning_budget: trace.reasoning_budget,
					  reasoning_content: trace.reasoning_content,
					  content: trace.content,
				  }
				: {
					  status: "unavailable",
					  endpoint: TRACE_ENDPOINT,
					  session_id: sessionId,
					  request_id: requestId,
					  error: "No matching Spark trace became visible before timeout",
				  };
			pi.appendEntry("qwen38fn-reasoning-trace", entry);
		} catch {
			// Do not turn a successful model response into an extension error.
		}
	});

	pi.registerEntryRenderer("qwen38fn-reasoning-trace", (entry, { expanded }, theme) => {
		const data = entry.data as TraceEntry;
		const budget = data.reasoning_budget;
		let text = theme.fg("accent", "Qwen38fn trace ") + theme.fg("muted", data.status);
		if (budget) {
			text += ` · ${budget.reasoning_tokens ?? "?"}/${budget.budget_tokens ?? "?"} reasoning tokens`;
			text += ` · ${budget.termination ?? "unknown"}`;
		}
		if (expanded) {
			text += `\n${theme.fg("dim", JSON.stringify(data, null, 2))}`;
		}
		return new Text(text, 0, 0);
	});
}

function summarizeTrace(trace: TraceRecord): Record<string, unknown> {
	return {
		id: trace.id,
		timestamp_ms: trace.timestamp_ms,
		task_id: trace.task_id,
		completion_id: trace.completion_id,
		session_id: trace.session_id,
		request_id: trace.request_id,
		prompt_tokens: trace.prompt_tokens,
		completion_tokens: trace.completion_tokens,
		reasoning_budget: trace.reasoning_budget,
		reasoning_chars: trace.reasoning_content?.length ?? 0,
		content_chars: trace.content?.length ?? 0,
		prompt_chars: trace.prompt?.length ?? 0,
	};
}

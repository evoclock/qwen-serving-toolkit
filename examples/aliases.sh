#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-only
# Example aliases after setting MODELCTL_CONFIG to a local profile file.
# These are intentionally not installed automatically.
export MODELCTL_CONFIG="${MODELCTL_CONFIG:-$HOME/.config/modelctl/qwen.env}"
alias model-start='modelctl start'
alias model-stop='modelctl stop'
alias model-status='modelctl status'
alias model-logs='modelctl logs'
alias model-clear-cache='modelctl clear-cache'

// User-facing rendering of proxy failures. Switches on the stable error
// envelope `code` (docs/API.md "Versioning and error codes" — closed set),
// never on the human `detail`, which may change between proxy versions.
//
// Deliberately vscode-free so it runs under plain vitest; the caller
// (chatView) decides which surface shows the message and which action
// button (if any) to attach.

import { AtlasApiError } from '../client/atlasClient';

/** What the UI should do with a failure beyond showing the message. */
export type ErrorAction = 'set-token' | 'none';

export interface RenderedError {
	/** One-line, user-facing. Always includes the proxy detail when present. */
	message: string;
	action: ErrorAction;
	/** True for failures worth a native notification, not just a chat card
	 * (auth and availability problems the user must fix outside the chat). */
	prominent: boolean;
}

/** Per-code message prefixes. Codes absent here fall through to the raw
 * `code: detail` rendering — forward-compatible with new proxy codes. */
const CODE_MESSAGES: Record<string, { prefix: string; action: ErrorAction; prominent: boolean }> = {
	unauthorized: { prefix: 'The ATLAS proxy rejected the request (unauthorized)', action: 'set-token', prominent: true },
	invalid_input: { prefix: 'The proxy rejected the request as invalid', action: 'none', prominent: false },
	unsupported_operation: { prefix: 'The proxy does not support this operation', action: 'none', prominent: false },
	permission_denied: { prefix: 'The proxy denied the operation', action: 'none', prominent: false },
	timeout: { prefix: 'The operation timed out on the proxy', action: 'none', prominent: false },
	cancelled: { prefix: 'The operation was cancelled', action: 'none', prominent: false },
	dependency_unavailable: {
		prefix: 'A proxy dependency is unavailable (model, lens, sandbox, or v3 service)',
		action: 'none',
		prominent: true,
	},
	incompatible_artifact: { prefix: 'Lens/ASA artifacts do not match the loaded model', action: 'none', prominent: false },
	resource_limit: { prefix: 'The proxy hit a resource limit', action: 'none', prominent: false },
	sandbox_policy_rejected: { prefix: 'The sandbox policy rejected the operation', action: 'none', prominent: false },
	model_failure: { prefix: 'The model failed to produce a usable response', action: 'none', prominent: false },
	internal_error: { prefix: 'The proxy hit an internal error', action: 'none', prominent: false },
};

/** Render any turn/request failure to a user-facing message + action. */
export function renderError(error: unknown): RenderedError {
	if (error instanceof AtlasApiError) {
		const known = CODE_MESSAGES[error.code];
		if (known) {
			const detail = error.detail && error.detail !== error.code ? `: ${error.detail}` : '.';
			return { message: `${known.prefix}${detail}`, action: known.action, prominent: known.prominent };
		}
		// Unknown code (newer proxy) — show it verbatim so it is reportable.
		const label = error.code || `HTTP ${error.status}`;
		return { message: `${label}: ${error.detail || 'request failed'}`, action: 'none', prominent: false };
	}
	if (error instanceof Error) {
		return { message: `Could not reach the ATLAS proxy: ${error.message}`, action: 'none', prominent: true };
	}
	return { message: `Could not reach the ATLAS proxy: ${String(error)}`, action: 'none', prominent: true };
}

/** Flatten anything throwable into one readable line for the output channel.
 *
 * Errors do not serialize: JSON.stringify(new Error('fetch failed')) is '{}'.
 * And "fetch failed" on its own says nothing — undici puts the reason on
 * `cause`, so a refused connection reads as `fetch failed (ECONNREFUSED)`
 * and a bad hostname as `fetch failed (ENOTFOUND)`. Without this the user
 * gets a message that cannot distinguish "proxy is down" from "wrong URL"
 * from "TLS rejected".
 */
export function describeForLog(detail: unknown): string {
	if (detail instanceof Error) {
		const parts = [`${detail.name}: ${detail.message}`];
		const cause = (detail as { cause?: unknown }).cause;
		if (cause instanceof Error) {
			const code = (cause as { code?: string }).code;
			parts.push(`cause=${code ?? cause.name}: ${cause.message}`);
		} else if (cause !== undefined) {
			parts.push(`cause=${String(cause)}`);
		}
		return parts.join(' | ');
	}
	if (typeof detail === 'string') {
		return detail;
	}
	try {
		return JSON.stringify(detail) ?? String(detail);
	} catch {
		return String(detail);
	}
}

// Unit tests for renderError: known-code mapping, unknown-code passthrough,
// AbortError exclusion (handled upstream), and network-failure rendering.

import { describe, expect, it } from 'vitest';
import { AtlasApiError } from '../src/client/atlasClient';
import { describeForLog, renderError } from '../src/util/errors';

function apiError(code: string, detail: string, status = 400): AtlasApiError {
	return new AtlasApiError(status, { error: code, detail }, detail);
}

describe('renderError', () => {
	it('maps unauthorized to a prominent set-token action', () => {
		const rendered = renderError(apiError('unauthorized', 'missing bearer token', 401));
		expect(rendered.action).toBe('set-token');
		expect(rendered.prominent).toBe(true);
		expect(rendered.message).toContain('unauthorized');
		expect(rendered.message).toContain('missing bearer token');
	});

	it('maps dependency_unavailable to a prominent message without action', () => {
		const rendered = renderError(apiError('dependency_unavailable', 'lens service down', 503));
		expect(rendered.action).toBe('none');
		expect(rendered.prominent).toBe(true);
		expect(rendered.message).toContain('lens service down');
	});

	it('renders every closed-set code with a friendly prefix', () => {
		const codes = [
			'invalid_input',
			'unsupported_operation',
			'permission_denied',
			'timeout',
			'cancelled',
			'incompatible_artifact',
			'resource_limit',
			'sandbox_policy_rejected',
			'model_failure',
			'internal_error',
		];
		for (const code of codes) {
			const rendered = renderError(apiError(code, 'detail text'));
			// Friendly prefix, not the raw code alone.
			expect(rendered.message).not.toMatch(new RegExp(`^${code}:`));
			expect(rendered.message).toContain('detail text');
			expect(rendered.action).toBe('none');
			expect(rendered.prominent).toBe(false);
		}
	});

	it('omits the duplicate detail when it matches the code', () => {
		const rendered = renderError(apiError('timeout', 'timeout'));
		expect(rendered.message.endsWith('.')).toBe(true);
		expect(rendered.message).not.toContain(': timeout');
	});

	it('passes unknown codes through verbatim (newer proxy)', () => {
		const rendered = renderError(apiError('quota_exceeded', 'monthly cap hit'));
		expect(rendered.message).toBe('quota_exceeded: monthly cap hit');
		expect(rendered.action).toBe('none');
	});

	it('labels a code-less HTTP failure by status', () => {
		const rendered = renderError(new AtlasApiError(502, {}, 'bad gateway'));
		expect(rendered.message).toContain('HTTP 502');
		expect(rendered.message).toContain('bad gateway');
	});

	it('renders plain Errors as unreachable-proxy (prominent)', () => {
		const rendered = renderError(new Error('fetch failed'));
		expect(rendered.message).toContain('Could not reach the ATLAS proxy');
		expect(rendered.message).toContain('fetch failed');
		expect(rendered.prominent).toBe(true);
	});

	it('stringifies non-Error throwables', () => {
		const rendered = renderError('boom');
		expect(rendered.message).toContain('boom');
	});
});

describe('describeForLog', () => {
	it('keeps an Error readable where JSON.stringify would not', () => {
		// This is the whole reason it exists.
		expect(JSON.stringify(new Error('fetch failed'))).toBe('{}');
		expect(describeForLog(new Error('fetch failed'))).toContain('fetch failed');
	});

	it('surfaces the undici cause code, which is the actionable part', () => {
		// "fetch failed" cannot distinguish a dead proxy from a wrong URL.
		// The cause can.
		const refused = new TypeError('fetch failed');
		(refused as { cause?: unknown }).cause = Object.assign(
			new Error('connect ECONNREFUSED 127.0.0.1:8090'), { code: 'ECONNREFUSED' });
		const line = describeForLog(refused);
		expect(line).toContain('fetch failed');
		expect(line).toContain('ECONNREFUSED');

		const dns = new TypeError('fetch failed');
		(dns as { cause?: unknown }).cause = Object.assign(
			new Error('getaddrinfo ENOTFOUND nope'), { code: 'ENOTFOUND' });
		expect(describeForLog(dns)).toContain('ENOTFOUND');
	});

	it('handles a cause that is not an Error', () => {
		const e = new Error('boom');
		(e as { cause?: unknown }).cause = 'plain string cause';
		expect(describeForLog(e)).toContain('plain string cause');
	});

	it('passes strings through and survives unserializable input', () => {
		expect(describeForLog('already a string')).toBe('already a string');
		const cyclic: Record<string, unknown> = {};
		cyclic.self = cyclic;
		expect(() => describeForLog(cyclic)).not.toThrow();
		expect(describeForLog(undefined)).toBeTypeOf('string');
	});
});

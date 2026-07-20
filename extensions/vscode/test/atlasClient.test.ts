// Integration tests for AtlasClient against the mockProxy fixture:
// happy-path turn, bearer auth, permission pause + resume, permission
// 404-as-success, error envelope, cancel/abort, /ready 200 and 503.

import { afterEach, describe, expect, it } from 'vitest';
import { agentDispatcher, AtlasApiError, AtlasClient } from '../src/client/atlasClient';
import type { AgentRequest, ChatEvent } from '../src/client/types';
import { MockProxy } from './fixtures/mockProxy';

const BASE_REQUEST: AgentRequest = {
	message: 'hello',
	working_dir: '.',
	mode: 'default',
	session_id: 'test-abc123',
};

let proxy: MockProxy | undefined;

afterEach(async () => {
	await proxy?.stop();
	proxy = undefined;
});

async function startProxy(options: ConstructorParameters<typeof MockProxy>[0] = {}): Promise<MockProxy> {
	proxy = new MockProxy(options);
	await proxy.start();
	return proxy;
}

describe('AtlasClient.sendAgentTurn', () => {
	it('streams a happy-path turn to completion', async () => {
		const mock = await startProxy({
			agentEvents: [
				{ type: 'turn_start', data: { turn: 1, messages: 1, trimmed: false } },
				{ type: 'text', data: { content: 'hi there' } },
				{ type: 'done', data: { summary: '' } },
			],
		});
		const client = new AtlasClient({ baseUrl: mock.url });

		const events: ChatEvent[] = [];
		for await (const event of client.sendAgentTurn(BASE_REQUEST)) {
			events.push(event);
		}
		expect(events.map((e) => e.type)).toEqual(['turn_start', 'text', 'done']);
		expect(mock.requests[0].body).toMatchObject({ message: 'hello', session_id: 'test-abc123' });
	});

	it('sends the bearer token on every request', async () => {
		const mock = await startProxy({ agentEvents: [] });
		const client = new AtlasClient({ baseUrl: mock.url, token: 'sk-test' });

		for await (const _ of client.sendAgentTurn(BASE_REQUEST)) {
			// drain
		}
		await client.cancelTurn('test-abc123');
		expect(mock.requests).toHaveLength(2);
		for (const request of mock.requests) {
			expect(request.headers.authorization).toBe('Bearer sk-test');
		}
	});

	it('omits the Authorization header when no token is set', async () => {
		const mock = await startProxy({ agentEvents: [] });
		const client = new AtlasClient({ baseUrl: mock.url });
		for await (const _ of client.sendAgentTurn(BASE_REQUEST)) {
			// drain
		}
		expect(mock.requests[0].headers.authorization).toBeUndefined();
	});

	it('throws AtlasApiError with the envelope code on non-200', async () => {
		const mock = await startProxy({
			agentError: { status: 401, body: { error: 'unauthorized', detail: 'bad token', api_version: '1.0.0' } },
		});
		const client = new AtlasClient({ baseUrl: mock.url });

		const iterate = async () => {
			for await (const _ of client.sendAgentTurn(BASE_REQUEST)) {
				// unreachable
			}
		};
		await expect(iterate()).rejects.toSatisfy((error: unknown) => {
			expect(error).toBeInstanceOf(AtlasApiError);
			const apiError = error as AtlasApiError;
			expect(apiError.status).toBe(401);
			expect(apiError.code).toBe('unauthorized');
			expect(apiError.detail).toBe('bad token');
			return true;
		});
	});

	it('keeps the code empty on a non-JSON error body', async () => {
		const mock = await startProxy({ agentError: { status: 502, body: 'Bad Gateway' } });
		const client = new AtlasClient({ baseUrl: mock.url });
		const iterate = async () => {
			for await (const _ of client.sendAgentTurn(BASE_REQUEST)) {
				// unreachable
			}
		};
		await expect(iterate()).rejects.toSatisfy((error: unknown) => {
			const apiError = error as AtlasApiError;
			expect(apiError.status).toBe(502);
			expect(apiError.code).toBe('');
			return true;
		});
	});

	it('aborts the stream via AbortSignal', async () => {
		const mock = await startProxy({
			agentEvents: [
				{ type: 'text', data: { content: 'one' } },
				{ type: 'text', data: { content: 'two' } },
				{ type: 'text', data: { content: 'three' } },
			],
			frameDelayMs: 50,
		});
		const client = new AtlasClient({ baseUrl: mock.url });
		const controller = new AbortController();

		const events: ChatEvent[] = [];
		const iterate = async () => {
			for await (const event of client.sendAgentTurn(BASE_REQUEST, controller.signal)) {
				events.push(event);
				controller.abort(); // abort after the first event
			}
		};
		await expect(iterate()).rejects.toThrow();
		expect(events).toHaveLength(1);
	});

	it('resumes after a permission pause when the decision is posted', async () => {
		const mock = await startProxy({
			agentEvents: [
				{
					type: 'permission_request',
					data: { tool_name: 'edit_file', args: {}, message: 'edit app.py?', tool_call_id: 'call_0' },
				},
				{ type: 'tool_result', data: { tool: 'edit_file', success: true, data: {}, elapsed: '10ms' } },
				{ type: 'done', data: { summary: '' } },
			],
			pauseAfterIndex: 0,
		});
		const client = new AtlasClient({ baseUrl: mock.url });

		const events: ChatEvent[] = [];
		for await (const event of client.sendAgentTurn(BASE_REQUEST)) {
			events.push(event);
			if (event.type === 'permission_request') {
				// Answer mid-stream, like the real UI flow.
				await client.postPermissionDecision({
					session_id: BASE_REQUEST.session_id,
					tool_call_id: 'call_0',
					decision: 'allow',
					scope: 'once',
				});
			}
		}
		expect(events.map((e) => e.type)).toEqual(['permission_request', 'tool_result', 'done']);
		const permissionPost = mock.requests.find((r) => r.url === '/v1/permission');
		expect(permissionPost?.body).toEqual({
			session_id: 'test-abc123',
			tool_call_id: 'call_0',
			decision: 'allow',
			scope: 'once',
		});
	});
});

describe('AtlasClient.postPermissionDecision', () => {
	it('treats 404 (already resolved) as success', async () => {
		const mock = await startProxy({ permissionStatus: 404 });
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(
			client.postPermissionDecision({
				session_id: 's',
				tool_call_id: 'call_0',
				decision: 'allow',
				scope: 'session',
			}),
		).resolves.toBeUndefined();
	});

	it('throws on other failures', async () => {
		const mock = await startProxy({ permissionStatus: 500 });
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(
			client.postPermissionDecision({ session_id: 's', tool_call_id: 'c', decision: 'deny', scope: 'once' }),
		).rejects.toBeInstanceOf(AtlasApiError);
	});
});

describe('AtlasClient.cancelTurn', () => {
	it('returns true when the proxy cancels', async () => {
		const mock = await startProxy();
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(client.cancelTurn('test-abc123')).resolves.toBe(true);
		expect(mock.requests[0].body).toEqual({ session_id: 'test-abc123' });
	});

	it('is best-effort: returns false on connection failure', async () => {
		// Port 1 is unassignable — connection refused.
		const client = new AtlasClient({ baseUrl: 'http://127.0.0.1:1' });
		await expect(client.cancelTurn('s')).resolves.toBe(false);
	});

	it('returns false for an empty session id without a request', async () => {
		const mock = await startProxy();
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(client.cancelTurn('')).resolves.toBe(false);
		expect(mock.requests).toHaveLength(0);
	});
});

describe('AtlasClient.getReady', () => {
	it('returns the gate body on 200', async () => {
		const mock = await startProxy();
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(client.getReady()).resolves.toMatchObject({ ready: true });
	});

	it('returns the gate body on 503 (degraded)', async () => {
		const mock = await startProxy({
			ready: { status: 503, body: { ready: false, inference: true, lens_ready: false, sandbox: true, v3: true } },
		});
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(client.getReady()).resolves.toMatchObject({ ready: false, lens_ready: false });
	});
});

describe('AtlasClient.getVersion', () => {
	it('returns version info', async () => {
		const mock = await startProxy();
		const client = new AtlasClient({ baseUrl: mock.url });
		await expect(client.getVersion()).resolves.toMatchObject({ api_version: '1.0.0', protocol_version: 1 });
	});
});

describe('AtlasClient base URL handling', () => {
	it('strips trailing slashes from the base URL', async () => {
		const mock = await startProxy();
		const client = new AtlasClient({ baseUrl: `${mock.url}///` });
		await client.getVersion();
		expect(mock.requests[0].url).toBe('/version');
	});
});

describe('agentDispatcher', () => {
	it('builds a dispatcher with the body timeout disabled on Node', async () => {
		// Node's fetch is undici-backed, so the global-dispatcher symbol must
		// exist and yield a constructible Agent. On a non-undici runtime this
		// would be undefined (graceful degradation) — but the extension host
		// and CI are both Node, so assert the strong case.
		const dispatcher = await agentDispatcher();
		expect(dispatcher).toBeTypeOf('object');
		// Memoized: the second call returns the same instance.
		await expect(agentDispatcher()).resolves.toBe(dispatcher);
	});

	it('streams a full turn with the dispatcher attached', async () => {
		// The happy-path turn above exercises sendAgentTurn, which now always
		// passes the dispatcher — re-assert explicitly for clarity.
		const mock = await startProxy({
			agentEvents: [
				{ type: 'text', data: { content: 'ok' } },
				{ type: 'done', data: { summary: '' } },
			],
		});
		const client = new AtlasClient({ baseUrl: mock.url });
		const events: ChatEvent[] = [];
		for await (const event of client.sendAgentTurn(BASE_REQUEST)) {
			events.push(event);
		}
		expect(events.at(-1)?.type).toBe('done');
	});
});

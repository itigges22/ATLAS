// Unit tests for the pure SSE frame parser. Coverage per plan: comments,
// chunk splits mid-frame, [DONE], >1MB frames, malformed frames, CRLF.

import { describe, expect, it } from 'vitest';
import { parseSSEStream } from '../src/client/sse';
import type { ChatEvent } from '../src/client/types';

const encoder = new TextEncoder();

/** Turn strings into an async byte stream, one chunk per string. */
async function* chunks(...parts: string[]): AsyncGenerator<Uint8Array> {
	for (const part of parts) {
		yield encoder.encode(part);
	}
}

async function collect(stream: AsyncIterable<Uint8Array>): Promise<ChatEvent[]> {
	const events: ChatEvent[] = [];
	for await (const event of parseSSEStream(stream)) {
		events.push(event);
	}
	return events;
}

function frame(event: ChatEvent): string {
	return `data: ${JSON.stringify(event)}\n\n`;
}

describe('parseSSEStream', () => {
	it('parses a simple stream and stops at [DONE]', async () => {
		const events = await collect(
			chunks(
				': connected\n\n',
				frame({ type: 'text', data: { content: 'hello' } }),
				frame({ type: 'done', data: { summary: '' } }),
				'data: [DONE]\n\n',
			),
		);
		expect(events).toEqual([
			{ type: 'text', data: { content: 'hello' } },
			{ type: 'done', data: { summary: '' } },
		]);
	});

	it('skips comments and blank lines', async () => {
		const events = await collect(
			chunks(': connected\n\n: heartbeat\n\n', frame({ type: 'text', data: { content: 'x' } }), 'data: [DONE]\n\n'),
		);
		expect(events).toHaveLength(1);
	});

	it('reassembles frames split across arbitrary chunk boundaries', async () => {
		const full = frame({ type: 'tool_call', data: { name: 'read_file', args: { path: 'a.py' }, turn: 1 } });
		// Split mid-"data:", mid-JSON, and mid-newline.
		const events = await collect(chunks('da', 'ta: {"type":"tool_call","da', full.slice(full.indexOf('"da') + 3), 'data: [DONE]\n\n'));
		expect(events).toEqual([{ type: 'tool_call', data: { name: 'read_file', args: { path: 'a.py' }, turn: 1 } }]);
	});

	it('handles a multibyte character split across chunks', async () => {
		const bytes = encoder.encode(frame({ type: 'text', data: { content: 'héllo→' } }) + 'data: [DONE]\n\n');
		// Split inside the é (2-byte UTF-8 sequence).
		const splitAt = 25;
		async function* twoChunks(): AsyncGenerator<Uint8Array> {
			yield bytes.slice(0, splitAt);
			yield bytes.slice(splitAt);
		}
		const events = await collect(twoChunks());
		expect(events).toEqual([{ type: 'text', data: { content: 'héllo→' } }]);
	});

	it('parses a >1MB frame', async () => {
		const big = 'x'.repeat(1_200_000);
		const events = await collect(
			chunks(frame({ type: 'tool_result', data: { tool: 'read_file', success: true, data: big } }), 'data: [DONE]\n\n'),
		);
		expect(events).toHaveLength(1);
		expect((events[0].data as { data: string }).data).toHaveLength(1_200_000);
	});

	it('skips malformed JSON frames without killing the stream', async () => {
		const events = await collect(
			chunks('data: {not json}\n\n', frame({ type: 'text', data: { content: 'ok' } }), 'data: [DONE]\n\n'),
		);
		expect(events).toEqual([{ type: 'text', data: { content: 'ok' } }]);
	});

	it('skips frames with a missing or empty type', async () => {
		const events = await collect(
			chunks('data: {"data":{"content":"no type"}}\n\n', 'data: {"type":"","data":{}}\n\n', 'data: [DONE]\n\n'),
		);
		expect(events).toEqual([]);
	});

	it('handles CRLF line endings', async () => {
		const events = await collect(
			chunks(': connected\r\n\r\ndata: {"type":"text","data":{"content":"crlf"}}\r\n\r\ndata: [DONE]\r\n\r\n'),
		);
		expect(events).toEqual([{ type: 'text', data: { content: 'crlf' } }]);
	});

	it('yields events already received when the stream ends without [DONE]', async () => {
		const events = await collect(chunks(frame({ type: 'text', data: { content: 'partial' } })));
		expect(events).toEqual([{ type: 'text', data: { content: 'partial' } }]);
	});

	it('flushes a final unterminated data line at end of stream', async () => {
		const events = await collect(chunks('data: {"type":"text","data":{"content":"tail"}}'));
		expect(events).toEqual([{ type: 'text', data: { content: 'tail' } }]);
	});

	it('ignores events after [DONE]', async () => {
		const events = await collect(
			chunks('data: [DONE]\n\n', frame({ type: 'text', data: { content: 'late' } })),
		);
		expect(events).toEqual([]);
	});
});

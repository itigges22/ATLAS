// The failure: "Please list the files in the directory you are in." returned
// a green chip, a duration, and no files. The proxy had sent the listing in
// `data` and the extension dropped it.

import { describe, expect, it } from 'vitest';
import { summarizeToolResult } from '../src/session/toolSummary';

/** tool_result as the proxy sends it: the tool's own output nested in `data`. */
function payload(data: unknown) {
	return { tool: 'list_directory', success: true, data };
}

describe('summarizeToolResult', () => {
	it('answers a directory listing with the entries', () => {
		const s = summarizeToolResult('list_directory', payload({
			entries: [
				{ name: 'app.py', type: 'file' },
				{ name: 'src', type: 'dir' },
			],
		}))!;
		expect(s).toContain('2 entries');
		expect(s).toContain('app.py');
		expect(s).toContain('src/'); // directories marked
	});

	it('caps a large listing instead of flooding the chip', () => {
		const entries = Array.from({ length: 40 }, (_, i) => ({ name: `f${i}.py`, type: 'file' }));
		const s = summarizeToolResult('list_directory', payload({ entries }))!;
		expect(s).toContain('40 entries');
		expect(s).toContain('+28 more');
		expect(s.split('\n')).toHaveLength(1);
	});

	it('says so when a directory is empty rather than rendering nothing', () => {
		expect(summarizeToolResult('list_directory', payload({ entries: [] }))).toBe('empty directory');
	});

	it('counts search matches', () => {
		expect(summarizeToolResult('search_files', payload({ matches: [1, 2, 3] }))).toBe('3 matches');
		expect(summarizeToolResult('search_files', payload({ matches: [1] }))).toBe('1 match');
		expect(summarizeToolResult('search_files', payload({ matches: [] }))).toBe('no matches');
	});

	it('falls back to the generic keys the TUI already uses', () => {
		expect(summarizeToolResult('run_command', payload({ stdout: 'ok' }))).toBe('ok');
		expect(summarizeToolResult('read_file', payload({ content: 'a\nb\nc' }))).toBe('3 lines');
	});

	it('returns undefined for a shape it does not know', () => {
		// Degrades to the old chip rather than inventing a summary.
		expect(summarizeToolResult('future_tool', payload({ weird: true }))).toBeUndefined();
		expect(summarizeToolResult('x', undefined)).toBeUndefined();
		expect(summarizeToolResult('x', payload({ stdout: '   ' }))).toBeUndefined();
	});
});

// Unit tests for the workspace-mismatch heuristic: per-kind verdicts,
// path extraction, the detector's FIFO matching, one-shot warning, and
// that failed results / inconclusive verdicts never fire.

import { describe, expect, it } from 'vitest';
import {
	MismatchDetector,
	fileOpPaths,
	fileOpVerdict,
	type PathStat,
} from '../src/workspace/mismatch';

const absent: PathStat = { exists: false };
const at = (mtimeMs: number): PathStat => ({ exists: true, mtimeMs });

describe('fileOpPaths', () => {
	it('extracts path for path-shaped tools', () => {
		expect(fileOpPaths('write_file', { path: 'a.py', content: 'x' })).toEqual({ primary: 'a.py' });
		expect(fileOpPaths('edit_file', { path: 'b.py' })).toEqual({ primary: 'b.py' });
		expect(fileOpPaths('ast_edit', { path: 'c.py' })).toEqual({ primary: 'c.py' });
		expect(fileOpPaths('delete_file', { path: 'd.py' })).toEqual({ primary: 'd.py' });
	});

	it('extracts source + destination for move_file', () => {
		expect(fileOpPaths('move_file', { source: 'a.py', destination: 'lib/a.py' })).toEqual({
			primary: 'a.py',
			secondary: 'lib/a.py',
		});
	});

	it('returns undefined for non-file-op tools and malformed args', () => {
		expect(fileOpPaths('run_command', { command: 'ls' })).toBeUndefined();
		expect(fileOpPaths('read_file', { path: 'a.py' })).toBeUndefined();
		expect(fileOpPaths('write_file', { path: 42 })).toBeUndefined();
		expect(fileOpPaths('move_file', { source: 'a.py' })).toBeUndefined();
		expect(fileOpPaths('write_file', null)).toBeUndefined();
	});
});

describe('fileOpVerdict', () => {
	it('write: new file appearing is a match, staying absent is a mismatch', () => {
		expect(fileOpVerdict('write', absent, at(2))).toBe('match');
		expect(fileOpVerdict('write', absent, absent)).toBe('mismatch');
	});

	it('write: overwrite must bump mtime', () => {
		expect(fileOpVerdict('write', at(1), at(2))).toBe('match');
		expect(fileOpVerdict('write', at(1), at(1))).toBe('mismatch');
	});

	it('edit: changed mtime is a match, unchanged or vanished is a mismatch', () => {
		expect(fileOpVerdict('edit', at(1), at(2))).toBe('match');
		expect(fileOpVerdict('edit', at(1), at(1))).toBe('mismatch');
		expect(fileOpVerdict('edit', at(1), absent)).toBe('mismatch');
	});

	it('edit: file never present locally is inconclusive (odd path, not mount proof)', () => {
		expect(fileOpVerdict('edit', absent, at(2))).toBe('inconclusive');
		expect(fileOpVerdict('edit', absent, absent)).toBe('inconclusive');
	});

	it('delete: disappearance is a match, persistence is a mismatch, pre-absent is inconclusive', () => {
		expect(fileOpVerdict('delete', at(1), absent)).toBe('match');
		expect(fileOpVerdict('delete', at(1), at(1))).toBe('mismatch');
		expect(fileOpVerdict('delete', absent, absent)).toBe('inconclusive');
	});

	it('move: source->destination is a match, source unmoved is a mismatch', () => {
		expect(fileOpVerdict('move', at(1), absent, absent, at(2))).toBe('match');
		expect(fileOpVerdict('move', at(1), at(1), absent, absent)).toBe('mismatch');
		expect(fileOpVerdict('move', absent, absent, absent, absent)).toBe('inconclusive');
		// Both present (copy-like state / same-name overwrite) — don't guess.
		expect(fileOpVerdict('move', at(1), at(1), absent, at(2))).toBe('inconclusive');
	});

	it('treats missing mtimes as changed (conservative: no false mismatch)', () => {
		expect(fileOpVerdict('edit', { exists: true }, at(2))).toBe('match');
		expect(fileOpVerdict('write', at(1), { exists: true })).toBe('match');
	});
});

/** Detector driven by a scripted stat table keyed by path; each recordToolResult
 * consults the CURRENT table so tests mutate it between call and result. */
function detector(stats: Map<string, PathStat>) {
	let fired = 0;
	const d = new MismatchDetector({
		stat: async (p) => stats.get(p) ?? absent,
		onMismatch: () => {
			fired += 1;
		},
		settleMs: 0,
	});
	return { d, firedCount: () => fired };
}

describe('MismatchDetector', () => {
	it('does not fire when the write shows up locally', async () => {
		const stats = new Map<string, PathStat>();
		const { d, firedCount } = detector(stats);
		await d.recordToolCall('write_file', { path: 'a.py', content: 'x' });
		stats.set('a.py', at(1)); // file appeared
		await d.recordToolResult('write_file', true);
		expect(firedCount()).toBe(0);
	});

	it('fires once when a successful write never lands locally', async () => {
		const stats = new Map<string, PathStat>();
		const { d, firedCount } = detector(stats);
		await d.recordToolCall('write_file', { path: 'a.py', content: 'x' });
		await d.recordToolResult('write_file', true);
		expect(firedCount()).toBe(1);
		// A second contradiction stays silent — the warning is one-shot.
		await d.recordToolCall('write_file', { path: 'b.py', content: 'y' });
		await d.recordToolResult('write_file', true);
		expect(firedCount()).toBe(1);
	});

	it('ignores failed results (nothing was applied to verify)', async () => {
		const stats = new Map<string, PathStat>();
		const { d, firedCount } = detector(stats);
		await d.recordToolCall('write_file', { path: 'a.py', content: 'x' });
		await d.recordToolResult('write_file', false);
		expect(firedCount()).toBe(0);
	});

	it('matches results to calls FIFO per tool name', async () => {
		const stats = new Map<string, PathStat>([
			['a.py', at(1)],
			['b.py', at(1)],
		]);
		const { d, firedCount } = detector(stats);
		await d.recordToolCall('edit_file', { path: 'a.py' });
		await d.recordToolCall('edit_file', { path: 'b.py' });
		stats.set('a.py', at(2)); // only the first edit landed
		await d.recordToolResult('edit_file', true); // a.py -> match
		expect(firedCount()).toBe(0);
		await d.recordToolResult('edit_file', true); // b.py unchanged -> mismatch
		expect(firedCount()).toBe(1);
	});

	it('stays silent on inconclusive verdicts', async () => {
		const stats = new Map<string, PathStat>();
		const { d, firedCount } = detector(stats);
		// edit of a file this workspace never had
		await d.recordToolCall('edit_file', { path: 'ghost.py' });
		await d.recordToolResult('edit_file', true);
		expect(firedCount()).toBe(0);
	});

	it('verifies move_file source/destination', async () => {
		const stats = new Map<string, PathStat>([['a.py', at(1)]]);
		const { d, firedCount } = detector(stats);
		await d.recordToolCall('move_file', { source: 'a.py', destination: 'lib/a.py' });
		stats.delete('a.py');
		stats.set('lib/a.py', at(2));
		await d.recordToolResult('move_file', true);
		expect(firedCount()).toBe(0);

		await d.recordToolCall('move_file', { source: 'lib/a.py', destination: 'lib/b.py' });
		// nothing moved locally
		await d.recordToolResult('move_file', true);
		expect(firedCount()).toBe(1);
	});

	it('recordDenied consumes the pending op so the next allowed call pairs correctly', async () => {
		const stats = new Map<string, PathStat>([
			['a.py', at(1)],
			['b.py', at(1)],
		]);
		const { d, firedCount } = detector(stats);
		// Denied edit: tool_call captured pre-op state, but the proxy emits
		// permission_denied and NO tool_result.
		await d.recordToolCall('edit_file', { path: 'a.py' });
		d.recordDenied('edit_file');
		// Allowed edit of b.py: must pair with ITS pre-op state, not a.py's.
		await d.recordToolCall('edit_file', { path: 'b.py' });
		stats.set('b.py', at(2));
		await d.recordToolResult('edit_file', true);
		expect(firedCount()).toBe(0);
	});

	it('recordDenied with nothing pending is a no-op', () => {
		const { d, firedCount } = detector(new Map());
		d.recordDenied('edit_file');
		expect(firedCount()).toBe(0);
	});

	it('reset drops pending pre-op state', async () => {
		const stats = new Map<string, PathStat>();
		const { d, firedCount } = detector(stats);
		await d.recordToolCall('write_file', { path: 'a.py', content: 'x' });
		d.reset();
		await d.recordToolResult('write_file', true); // no pending op to check
		expect(firedCount()).toBe(0);
	});
});

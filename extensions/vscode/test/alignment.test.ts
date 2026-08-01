// The failure this exists for: the proxy was bound to ~/snake-game while VS
// Code had ~/demo2 open. Every tool call was truthful and useless — the agent
// listed and edited a directory the user was not looking at. It took four
// rounds of manual `atlas workspace align` to notice each time.

import { describe, expect, it, vi } from 'vitest';
import { checkAlignment } from '../src/workspace/alignment';

describe('checkAlignment', () => {
	it('maps the `atlas workspace` exit code to a state', async () => {
		// 0 = this directory is covered by the bind, 1 = it is not.
		await expect(checkAlignment('/x', async () => 0)).resolves.toBe('aligned');
		await expect(checkAlignment('/x', async () => 1)).resolves.toBe('misaligned');
	});

	it('is unknown when atlas is not installed', async () => {
		// A missing CLI must never nag a user whose setup is otherwise fine.
		const spawnFailure = () => Promise.reject(new Error('ENOENT'));
		await expect(checkAlignment('/x', spawnFailure)).resolves.toBe('unknown');
	});

	it('is unknown on any other exit code', async () => {
		// e.g. docker unreachable — do not claim misalignment we cannot see.
		await expect(checkAlignment('/x', async () => 2)).resolves.toBe('unknown');
		await expect(checkAlignment('/x', async () => 127)).resolves.toBe('unknown');
	});

	it('runs in the folder being checked', async () => {
		const run = vi.fn(async () => 0);
		await checkAlignment('/home/isaac/demo2', run);
		expect(run).toHaveBeenCalledWith('atlas workspace', '/home/isaac/demo2');
	});
});

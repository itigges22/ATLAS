// Webview-side renderer for the ATLAS chat panel. Deliberately dumb: all
// state lives in the extension host (src/ui/chatView.ts); this script only
// renders the messages it is posted, in order. On load it sends {type:
// "ready"} and the host replays the full transcript.

/* global acquireVsCodeApi */

(function () {
	'use strict';

	const vscode = acquireVsCodeApi();

	const messagesEl = document.getElementById('messages');
	const inputEl = document.getElementById('input');
	const sendEl = document.getElementById('send');
	const stopEl = document.getElementById('stop');

	/** The assistant bubble currently receiving streamed text, if any. */
	let openAssistantEl = null;
	/** Tool chips awaiting a tool_result, keyed by tool name (FIFO per name). */
	const pendingChips = new Map();

	function scrollToBottom() {
		messagesEl.scrollTop = messagesEl.scrollHeight;
	}

	function appendBlock(className, text) {
		const el = document.createElement('div');
		el.className = className;
		if (text !== undefined) {
			el.textContent = text;
		}
		messagesEl.appendChild(el);
		scrollToBottom();
		return el;
	}

	function closeAssistantBubble() {
		openAssistantEl = null;
	}

	function assistantBubble() {
		if (!openAssistantEl) {
			openAssistantEl = appendBlock('msg assistant', '');
		}
		return openAssistantEl;
	}

	function addToolChip(name, detail) {
		const chip = document.createElement('div');
		chip.className = 'chip pending';
		const label = document.createElement('span');
		label.className = 'chip-label';
		label.textContent = detail ? name + ' ' + detail : name;
		const status = document.createElement('span');
		status.className = 'chip-status';
		status.textContent = '⋯'; // ⋯ spinner placeholder
		chip.appendChild(status);
		chip.appendChild(label);
		messagesEl.appendChild(chip);
		scrollToBottom();

		if (!pendingChips.has(name)) {
			pendingChips.set(name, []);
		}
		pendingChips.get(name).push(chip);
	}

	function resolveToolChip(tool, success, elapsed, error) {
		const queue = pendingChips.get(tool);
		const chip = queue && queue.length > 0 ? queue.shift() : null;
		if (!chip) {
			// Result without a rendered call (e.g. replay edge) — show standalone.
			appendBlock('chip ' + (success ? 'ok' : 'fail'), (success ? '✓ ' : '✗ ') + tool + (error ? ': ' + error : ''));
			return;
		}
		chip.classList.remove('pending');
		chip.classList.add(success ? 'ok' : 'fail');
		const status = chip.querySelector('.chip-status');
		status.textContent = success ? '✓' : '✗';
		if (elapsed) {
			const time = document.createElement('span');
			time.className = 'chip-elapsed';
			time.textContent = elapsed;
			chip.appendChild(time);
		}
		if (!success && error) {
			const detail = document.createElement('div');
			detail.className = 'chip-error';
			detail.textContent = error;
			chip.appendChild(detail);
		}
	}

	function setBusy(busy) {
		sendEl.hidden = busy;
		stopEl.hidden = !busy;
		inputEl.disabled = busy;
	}

	function resetAll() {
		messagesEl.textContent = '';
		pendingChips.clear();
		openAssistantEl = null;
	}

	window.addEventListener('message', (event) => {
		const message = event.data;
		switch (message.type) {
			case 'userMessage':
				closeAssistantBubble();
				appendBlock('msg user', message.text);
				break;
			case 'assistantDelta':
				assistantBubble().textContent += message.text;
				scrollToBottom();
				break;
			case 'toolCall':
				closeAssistantBubble();
				addToolChip(message.name, message.detail);
				break;
			case 'toolResult':
				resolveToolChip(message.tool, message.success, message.elapsed, message.error);
				break;
			case 'note':
				closeAssistantBubble();
				appendBlock('note', message.text);
				break;
			case 'turnDone':
				closeAssistantBubble();
				break;
			case 'turnError':
				closeAssistantBubble();
				appendBlock('error-card', message.message);
				break;
			case 'busy':
				setBusy(message.value);
				break;
			case 'reset':
				resetAll();
				break;
		}
	});

	function submit() {
		const text = inputEl.value.trim();
		if (text === '') {
			return;
		}
		inputEl.value = '';
		vscode.postMessage({ type: 'submit', text: text });
	}

	sendEl.addEventListener('click', submit);
	stopEl.addEventListener('click', () => vscode.postMessage({ type: 'cancel' }));
	inputEl.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			submit();
		}
	});

	vscode.postMessage({ type: 'ready' });
})();

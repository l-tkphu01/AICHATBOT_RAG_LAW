import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { sendChatMessage } from '../services/api.js'

const DEFAULT_STEPS = [
	{ key: 'check', label: 'Kiểm tra câu hỏi' },
	{ key: 'retrieve', label: 'Truy xuất căn cứ' },
	{ key: 'compose', label: 'Soạn câu trả lời' },
]

const makeId = () => {
	if (typeof crypto !== 'undefined' && crypto?.randomUUID) return crypto.randomUUID()
	return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const toHistory = (messages, maxTurns = 8) => {
	const trimmed = messages
		.filter((m) => m.role === 'user' || m.role === 'assistant')
		.slice(-maxTurns)
		.map((m) => ({ role: m.role, content: m.content }))
	return trimmed
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const streamSimulated = async ({ text, onDelta, isCancelled }) => {
	const chunks = text.split(/(\s+)/)
	for (let index = 0; index < chunks.length; index += 1) {
		if (isCancelled()) return
		onDelta(chunks[index])
		// nhanh hơn với khoảng trắng để mượt giống GPT
		const delay = chunks[index].trim() ? 18 : 6
		// eslint-disable-next-line no-await-in-loop
		await sleep(delay)
	}
}

export function useChat() {
	const [messages, setMessages] = useState([])
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState('')
	const [progress, setProgress] = useState({
		visible: false,
		steps: DEFAULT_STEPS,
		activeIndex: 0,
		percent: 0,
		label: '',
	})

	const abortRef = useRef(null)
	const cancelRef = useRef({ cancelled: false })
	const progressTimerRef = useRef(null)

	const reset = useCallback(() => {
		abortRef.current?.abort?.()
		cancelRef.current.cancelled = true
		setMessages([])
		setError('')
		setLoading(false)
		setProgress((prev) => ({ ...prev, visible: false, activeIndex: 0, percent: 0, label: '' }))
	}, [])

	const stopProgressTimer = useCallback(() => {
		if (progressTimerRef.current) {
			clearInterval(progressTimerRef.current)
			progressTimerRef.current = null
		}
	}, [])

	const startProgress = useCallback(() => {
		stopProgressTimer()

		setProgress((prev) => ({
			...prev,
			visible: true,
			activeIndex: 0,
			percent: 6,
			label: 'Đang kiểm tra câu hỏi…',
		}))

		const startedAt = Date.now()
		progressTimerRef.current = setInterval(() => {
			const elapsed = Date.now() - startedAt
			// 0-30%: check, 30-75%: retrieve, 75-96%: compose
			let nextPercent = 6
			let nextIndex = 0
			let nextLabel = 'Đang kiểm tra câu hỏi…'

			if (elapsed < 1500) {
				nextPercent = Math.min(30, 6 + (elapsed / 1500) * 24)
			} else if (elapsed < 4200) {
				nextIndex = 1
				nextLabel = 'Đang truy xuất căn cứ liên quan…'
				nextPercent = Math.min(75, 30 + ((elapsed - 1500) / 2700) * 45)
			} else {
				nextIndex = 2
				nextLabel = 'Đang soạn câu trả lời…'
				nextPercent = Math.min(96, 75 + ((elapsed - 4200) / 2600) * 21)
			}

			setProgress((prev) => ({ ...prev, activeIndex: nextIndex, percent: nextPercent, label: nextLabel }))
		}, 120)
	}, [stopProgressTimer])

	const finishProgress = useCallback(
		async (finalLabel) => {
			stopProgressTimer()
			setProgress((prev) => ({ ...prev, percent: 100, label: finalLabel || 'Hoàn tất.' }))
			await sleep(350)
			setProgress((prev) => ({ ...prev, visible: false }))
		},
		[stopProgressTimer],
	)

	const send = useCallback(
		async (question) => {
			const text = (question || '').trim()
			if (!text || loading) return

			setError('')
			setLoading(true)
			startProgress()

			abortRef.current?.abort?.()
			abortRef.current = new AbortController()
			cancelRef.current.cancelled = false

			const userMessage = { id: makeId(), role: 'user', content: text }
			const assistantId = makeId()
			const assistantMessage = {
				id: assistantId,
				role: 'assistant',
				content: '',
				meta: 'AI pháp lý',
				status: 'streaming',
				sources: [],
			}

			setMessages((prev) => [...prev, userMessage, assistantMessage])

			try {
				const history = toHistory([...messages, userMessage])
				const data = await sendChatMessage(text, { history, stream: true, signal: abortRef.current.signal })

				const status = data?.status || 'success'
				const answer = data?.answer || data?.response || 'Không nhận được phản hồi từ hệ thống.'
				const sources = Array.isArray(data?.sources) ? data.sources : []

				if (cancelRef.current.cancelled) return

				const meta =
					status === 'blocked'
						? 'Bị chặn bởi Guardian'
						: sources.length
							? `Có ${sources.length} căn cứ liên quan`
							: 'AI pháp lý'

				setMessages((prev) =>
					prev.map((m) =>
						m.id === assistantId
							? {
									...m,
									meta,
									status,
									sources,
									content: '',
								}
							: m,
					),
				)

				await finishProgress(status === 'blocked' ? 'Đã chặn yêu cầu.' : 'Đã nhận kết quả.')

				// Stream giả lập (backend hiện chưa stream thật)
				await streamSimulated({
					text: answer,
					isCancelled: () => cancelRef.current.cancelled,
					onDelta: (delta) => {
						setMessages((prev) =>
							prev.map((m) => (m.id === assistantId ? { ...m, content: `${m.content}${delta}` } : m)),
						)
					},
				})

				setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, status: 'done' } : m)))
			} catch (err) {
				if (err?.name === 'AbortError') return
				console.error(err)
				setError('Không thể kết nối tới backend. Hãy kiểm tra API hoặc thử lại sau.')
				await finishProgress('Có lỗi kết nối.')
				setMessages((prev) =>
					prev.map((m) =>
						m.id === assistantId
							? {
									...m,
									status: 'error',
									meta: 'Kết nối thất bại',
									content: 'Hiện tại tôi chưa lấy được kết quả từ máy chủ. Vui lòng thử lại sau.',
								}
							: m,
					),
				)
			} finally {
				setLoading(false)
			}
		},
		[finishProgress, loading, messages, startProgress],
	)

	const cancel = useCallback(() => {
		cancelRef.current.cancelled = true
		abortRef.current?.abort?.()
		stopProgressTimer()
		setLoading(false)
		setProgress((prev) => ({ ...prev, visible: false }))
	}, [stopProgressTimer])

	useEffect(() => () => cancel(), [cancel])

	const ui = useMemo(
		() => ({
			steps: progress.steps,
			progressPercent: progress.percent,
			progressLabel: progress.label,
			progressVisible: progress.visible,
			progressActiveIndex: progress.activeIndex,
		}),
		[progress],
	)

	return {
		messages,
		loading,
		error,
		reset,
		cancel,
		send,
		ui,
	}
}

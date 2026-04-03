const CHAT_ENDPOINT = '/api/v1/chat'

const buildError = async (response) => {
  try {
    const data = await response.json()
    const detail = typeof data?.detail === 'string' ? data.detail : ''
    return new Error(detail || `HTTP ${response.status}`)
  } catch {
    return new Error(`HTTP ${response.status}`)
  }
}

/**
 * @param {string} question
 * @param {{history?: Array<{role: string, content: string}>, stream?: boolean, signal?: AbortSignal}} [options]
 */
export const sendChatMessage = async (question, options = {}) => {
  const { history = [], stream = true, signal } = options

  const response = await fetch(CHAT_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    signal,
    body: JSON.stringify({
      query: question,
      history,
      stream,
    }),
  })

  if (!response.ok) {
    throw await buildError(response)
  }

  return await response.json()
}

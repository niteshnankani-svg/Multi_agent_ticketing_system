import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://54.91.16.86:8000'

const CATEGORY_LABELS = {
  finance: 'Finance',
  backend: 'Backend',
  internal: 'Internal',
  general: 'General',
  cached: 'Cached',
}

function CategoryBadge({ category }) {
  if (!category) return null
  const key = category.toLowerCase()
  const label = CATEGORY_LABELS[key] || category
  return <span className={`badge badge-${key}`}>{label}</span>
}

function App() {
  const [ticketText, setTicketText] = useState('')
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const canSubmit = ticketText.trim().length > 0 && status !== 'loading'

  async function handleSubmit(e) {
    e.preventDefault()
    if (!canSubmit) return

    setStatus('loading')
    setErrorMessage('')
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/ticket`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_text: ticketText }),
      })

      if (!res.ok) {
        let detail = `Request failed with status ${res.status}`
        try {
          const body = await res.json()
          if (body?.detail) detail = body.detail
        } catch {
          // response wasn't JSON, keep the generic message
        }
        throw new Error(detail)
      }

      const data = await res.json()
      setResult(data)
      setStatus('success')
    } catch (err) {
      setErrorMessage(
        err instanceof TypeError
          ? `Could not reach the API at ${API_URL}. It may be offline.`
          : err.message || 'Something went wrong.'
      )
      setStatus('error')
    }
  }

  const escalated =
    Boolean(result?.needs_human) || result?.status === 'needs_human_review'
  const confidence =
    typeof result?.confidence === 'number' ? result.confidence : null
  const category = result?.predicted_category || result?.category

  return (
    <div className="page">
      <header className="header">
        <h1>Ticket Router</h1>
        <p className="subtitle">
          Multi-agent classification &amp; response system
        </p>
      </header>

      <main className="layout">
        <form className="card" onSubmit={handleSubmit}>
          <label className="field-label" htmlFor="ticket-text">
            Ticket text
          </label>
          <textarea
            id="ticket-text"
            className="textarea"
            placeholder="Describe the issue, e.g. &quot;I was charged twice for my subscription this month.&quot;"
            value={ticketText}
            onChange={(e) => setTicketText(e.target.value)}
            rows={8}
            disabled={status === 'loading'}
          />
          <div className="form-footer">
            <span className="hint">POST {API_URL}/ticket</span>
            <button type="submit" className="submit-btn" disabled={!canSubmit}>
              {status === 'loading' ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Submitting…
                </>
              ) : (
                'Submit Ticket'
              )}
            </button>
          </div>
        </form>

        {status === 'error' && (
          <div className="card result-card error-card">
            <h2 className="result-heading">Request failed</h2>
            <p className="error-text">{errorMessage}</p>
          </div>
        )}

        {status === 'success' && result && (
          <div className={`card result-card ${escalated ? 'escalated' : ''}`}>
            {escalated && (
              <div className="escalation-banner">
                <span className="escalation-dot" aria-hidden="true" />
                Escalated for human review
              </div>
            )}

            <div className="result-meta">
              <CategoryBadge category={category} />
              {confidence !== null && (
                <div className="confidence">
                  <span className="confidence-label">Confidence</span>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${Math.round(confidence * 100)}%` }}
                    />
                  </div>
                  <span className="confidence-value">
                    {Math.round(confidence * 100)}%
                  </span>
                </div>
              )}
              {result.thread_id && (
                <span className="thread-id" title="Thread ID">
                  {result.thread_id}
                </span>
              )}
            </div>

            {result.answer && (
              <div className="answer-block">
                <h2 className="result-heading">Answer</h2>
                <p className="answer-text">{result.answer}</p>
              </div>
            )}

            {escalated && result.escalation_reason && (
              <div className="reason-block">
                <h2 className="result-heading">Reason</h2>
                <p className="reason-text">{result.escalation_reason}</p>
              </div>
            )}

            {!result.answer && result.reason && (
              <div className="reason-block">
                <h2 className="result-heading">Reason</h2>
                <p className="reason-text">{result.reason}</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App

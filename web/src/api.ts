// Loose JSON shape: the backend is the contract (see src/membridge/api/app.py).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Rec = Record<string, any>;

export async function api<T = Rec>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const r = await fetch(path, {
    method: opts.method ?? "GET",
    headers: { "Content-Type": "application/json", "x-membridge": "1" },
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || data.detail || r.statusText);
  return data as T;
}

export const when = (t?: number | null) => (t ? new Date(t * 1000).toLocaleString() : "—");

export const CATEGORIES = [
  "preference", "decision", "project", "constraint", "profile", "skill", "fact",
  "lesson", "interest", "health", "education", "research", "session",
];

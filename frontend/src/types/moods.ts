export interface Mood {
  id: string
  name: string
  score: number
  emoji?: string | null
}

export interface MoodResponse {
  count: number
  next: string | null
  previous: string | null
  results: Mood[]
}

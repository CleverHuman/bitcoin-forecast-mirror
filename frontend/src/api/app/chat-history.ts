import { Message } from '@/types/message'

interface ChatHistory {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
}

const CHAT_HISTORY_KEY = 'chat_history'

export function saveChat(messages: Message[]): string {
  const history = getChatHistory()
  const id = crypto.randomUUID()
  const title = messages[0]?.content.slice(0, 50) || 'New Chat'
  
  const newChat: ChatHistory = {
    id,
    title,
    messages,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }

  history.unshift(newChat)
  localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history))
  return id
}

export function updateChat(id: string, messages: Message[]) {
  const history = getChatHistory()
  const index = history.findIndex(chat => chat.id === id)
  
  if (index !== -1) {
    history[index] = {
      ...history[index],
      messages,
      updatedAt: new Date().toISOString()
    }
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history))
  }
}

export function getChatHistory(): ChatHistory[] {
  const historyStr = localStorage.getItem(CHAT_HISTORY_KEY)
  return historyStr ? JSON.parse(historyStr) : []
}

export function getChatById(id: string): ChatHistory | undefined {
  const history = getChatHistory()
  return history.find(chat => chat.id === id)
}

export function deleteChat(id: string) {
  const history = getChatHistory()
  const filtered = history.filter(chat => chat.id !== id)
  localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(filtered))
}
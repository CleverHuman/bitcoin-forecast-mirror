import { Message } from '@/types/message'
import OpenAI from 'openai'

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export async function getChatCompletion(messages: ChatMessage[], onUpdate?: (content: string) => void): Promise<Message> {
  try {
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY
    if (!apiKey) {
      throw new Error('OpenAI API key is not configured')
    }

    const openai = new OpenAI({
      apiKey: apiKey,
      dangerouslyAllowBrowser: true
    })

    const stream = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      temperature: 0.7,
      stream: true,
    })

    let fullContent = ''
    let responseId = ''
    let role = 'assistant'

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || ''
      fullContent += content
      if (!responseId && chunk.id) {
        responseId = chunk.id
      }
      if (chunk.choices[0]?.delta?.role) {
        role = chunk.choices[0].delta.role
      }
      if (onUpdate) {
        onUpdate(fullContent)
      }
    }

    return {
      id: responseId,
      content: fullContent,
      role: role as 'assistant'
    }
  } catch (error) {
    console.error('Error calling OpenAI API:', error)
    throw new Error('Failed to get AI response')
  }
}
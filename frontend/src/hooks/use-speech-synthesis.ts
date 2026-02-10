import { useState, useEffect } from 'react'

interface UseSpeechSynthesisProps {
  onEnd?: () => void
}

export function useSpeechSynthesis({ onEnd }: UseSpeechSynthesisProps = {}) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [utterance, setUtterance] = useState<SpeechSynthesisUtterance | null>(null)

  useEffect(() => {
    return () => {
      if (utterance) {
        window.speechSynthesis.cancel()
      }
    }
  }, [utterance])

  const play = (content?: string) => {
    const textToSpeak = content?.trim()
    if (!textToSpeak) {
      console.warn('No text content provided for speech synthesis')
      return
    }

    if (isPlaying) {
      window.speechSynthesis.cancel()
      setIsPlaying(false)
      setUtterance(null)
      return
    }

    const newUtterance = new SpeechSynthesisUtterance(textToSpeak)

    const voices = window.speechSynthesis.getVoices()
    const englishVoice = voices.find(voice => voice.lang.startsWith('en-'))
    newUtterance.voice = englishVoice || voices[0]

    newUtterance.rate = 1.0
    newUtterance.pitch = 1.0
    newUtterance.volume = 1.0
    newUtterance.lang = 'en-US'

    newUtterance.onend = () => {
      setIsPlaying(false)
      setUtterance(null)
      onEnd?.()
    }

    newUtterance.onerror = () => {
      setIsPlaying(false)
      setUtterance(null)
    }

    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(newUtterance)
    setUtterance(newUtterance)
    setIsPlaying(true)
  }

  const stop = () => {
    if (isPlaying) {
      window.speechSynthesis.cancel()
      setIsPlaying(false)
      setUtterance(null)
    }
  }

  return {
    isPlaying,
    play,
    stop
  }
}
import { createContext, useContext, useEffect, useState, ReactNode } from "react"
import { getCoinsMarkets } from "@/api/coingecko/coin"
import { CryptoData } from "@/stores/slices/coingecko/coins-market"


interface CryptoContextType {
  cryptoData: CryptoData[]
  isLoading: boolean
  error: Error | null
}

const CryptoContext = createContext<CryptoContextType | undefined>(undefined)

const REFRESH_INTERVAL = 15 * 60 * 1000 

export function CryptoProvider({ children }: { children: ReactNode }) {
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true)
        const data = await getCoinsMarkets(1, 40, false)
        setCryptoData(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to fetch crypto data'))
      } finally {
        setIsLoading(false)
      }
    }

    fetchData() 
    const intervalId = setInterval(fetchData, REFRESH_INTERVAL)

    return () => clearInterval(intervalId)
  }, [])

  return (
    <CryptoContext.Provider value={{ cryptoData, isLoading, error }}>
      {children}
    </CryptoContext.Provider>
  )
}

export function useCrypto() {
  const context = useContext(CryptoContext)
  if (context === undefined) {
    throw new Error('useCrypto must be used within a CryptoProvider')
  }
  return context
} 
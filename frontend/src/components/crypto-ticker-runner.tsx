import { useEffect, useState } from "react"
import { motion, useAnimation, useMotionValue } from "framer-motion"
import { ArrowUpIcon, ArrowDownIcon, PlayIcon, PauseIcon } from "lucide-react"
import { useCrypto } from "@/contexts/crypto-context"
import { CryptoData } from "@/stores/slices/coingecko/coins-market"

export function CryptoTickerRunner() {
  const { cryptoData } = useCrypto()
  const controls = useAnimation()
  const x = useMotionValue(0) 
  const [isPaused, setIsPaused] = useState(false)
  const [currentX, setCurrentX] = useState(0)

  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsPaused(document.hidden)
    }

    document.addEventListener("visibilitychange", handleVisibilityChange)

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange)
    }
  }, [])

  useEffect(() => {
    if (isPaused) {
      controls.stop()
      setCurrentX(x.get()) 
    } else {
      controls.start({
        x: [currentX, "-50%"],
        transition: {
          ease: "linear",
          duration: (140), 
          repeat: Number.POSITIVE_INFINITY,
          repeatType: "loop",
          type: "tween",
        },
      })
    }
  }, [isPaused, controls, currentX, x])

  const togglePause = () => {
    if (!isPaused) {
      setCurrentX(x.get()) 
    }
    setIsPaused(!isPaused)
  }

  const handleTickerClick = (_crypto: CryptoData) => {
    // navigate(RouterPath.TOKENS + `/${crypto.id}`)
  }

  return (
    <div className="w-full overflow-hidden border-b relative">
      <div className="relative w-full h-8 left-4">
        <motion.div
          className="flex w-max absolute top-0"
          animate={controls}
          initial={{ x: currentX }}
          style={{ x }} // Bind x motion value
        >
          {cryptoData.concat(cryptoData).map((crypto, index) => (
            <div 
              key={`${crypto.id}-${index}`} 
              className="mx-2 flex items-center space-x-2 bg-background/50 rounded-md px-3 py-1.5 text-sm"
            >
              <span className="font-semibold text-primary cursor-pointer" onClick={() => handleTickerClick(crypto)}>{(crypto.symbol as string).toUpperCase()}</span>
              <span className="text-foreground/80">${crypto.current_price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              })}</span>
              <span
                className={`flex items-center ${
                  (crypto.price_change_percentage_24h as number)  >= 0 ? "text-green-500" : "text-red-500"
                }`}
              >
                {(crypto.price_change_percentage_24h as number) >= 0 ? (
                  <ArrowUpIcon className="h-3 w-3" />
                ) : (
                  <ArrowDownIcon className="h-3 w-3" />
                )}
                {Math.abs((crypto.price_change_percentage_24h as number) ).toFixed(2)}%
              </span>
            </div>
          ))}
        </motion.div>
      </div>
      <button onClick={togglePause} className="absolute top-0 left-0 p-2 bg-background text-sm">
        {isPaused ? <PlayIcon size={14} /> : <PauseIcon size={14} />}
      </button>
    </div>
  )
}

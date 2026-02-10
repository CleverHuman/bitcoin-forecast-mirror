import ThemeSwitch from "./theme-switch"
import { UserNav } from "./user-nav"
import { CryptoTickerRunner } from "./crypto-ticker-runner"
import { useLocation } from "react-router-dom"

function HeaderPage() {
  // const { isAuthenticated } = useAuth()
  const location = useLocation()
  const showTicker = location.pathname.startsWith('/feed')

  return (
    <div
      className={`relative top-0 z-50 flex w-full items-center justify-between bg-background pr-4 py-2 md:sticky`}
    >
      
      <div className='flex-1'>
        {showTicker && <CryptoTickerRunner />}
      </div>
      <div className='flex items-center space-x-4'>
        {/* {isAuthenticated && <TeamSwitcher />} */}
        <ThemeSwitch />
        <UserNav />
      </div>
    </div>
  )
}

export default HeaderPage

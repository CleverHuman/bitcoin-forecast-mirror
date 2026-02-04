import React from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import { ThemeProvider } from '@/components/theme-provider'
import router from '@/router/index'
import '@/index.css'
import { Provider } from 'react-redux'
import { store } from './stores/store'
import { BlockchainProvider } from './contexts/blockchain-context'
import { AuthProvider } from './contexts/auth-context'
import { SearchProvider } from './contexts/search-context'
import { UserProvider } from './contexts/user-context'
import { TeamProvider } from './contexts/team-context'
import { CryptoProvider } from './contexts/crypto-context'

const script = document.createElement('script')
script.src = 'http://localhost:8097'
document.head.appendChild(script)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <ThemeProvider defaultTheme='dark' storageKey='vite-ui-theme'>
        <AuthProvider>
          <BlockchainProvider>
            <CryptoProvider>
              <SearchProvider>
                <UserProvider>
                  <TeamProvider>
                    <RouterProvider router={router} />
                  </TeamProvider>
                </UserProvider>
              </SearchProvider>
            </CryptoProvider>
          </BlockchainProvider>
        </AuthProvider>
        <Toaster />
      </ThemeProvider>
    </Provider>
  </React.StrictMode>
)

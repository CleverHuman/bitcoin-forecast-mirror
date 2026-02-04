import { createBrowserRouter, Navigate } from 'react-router-dom'
import GeneralError from '../pages/errors/general-error.tsx'
import NotFoundError from '../pages/errors/not-found-error.tsx'
import MaintenanceError from '../pages/errors/maintenance-error.tsx'
import UnauthorisedError from '../pages/errors/unauthorised-error.tsx'
import { RouterPath } from './router-path.ts'
import AuthRoute from './auth-routes.tsx'
import AppShell from '@/components/app-shell.tsx'


const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <Navigate to={RouterPath.SIGN_IN} replace />,
    },

    {
      path: '/',
      element: <AuthRoute />,
      children: [
        {
          path: RouterPath.SIGN_IN.slice(1),
          lazy: async () => ({
            Component: (await import('../pages/auth/sign-in.tsx')).default,
          }),
        },
        {
          path: RouterPath.SIGN_UP,
          lazy: async () => ({
            Component: (await import('../pages/auth/sign-up.tsx')).default,
          }),
        },
        {
          path: RouterPath.FORGOT_PASSWORD,
          lazy: async () => ({
            Component: (await import('../pages/auth/forgot-password.tsx'))
              .default,
          }),
        },
        {
          path: RouterPath.OTP,
          lazy: async () => ({
            Component: (await import('../pages/auth/otp.tsx')).default,
          }),
        },
      ],
    },

    {
      element: <AppShell />,
      errorElement: <GeneralError />,
      children: [
        {
          index: true,
          path: RouterPath.HOME.slice(1),
          lazy: async () => ({
            Component: (await import('../pages/index.tsx')).default,
          }),
        },
        {
          path: RouterPath.FORECASTING,
          lazy: async () => ({
            Component: (await import('../pages/forecasting/index.tsx')).default,
          }),
        },
      ],
    },

    {
      path: '*',
      element: <Navigate to={RouterPath.ERROR_404} replace />,
    },
    {
      path: RouterPath.ERROR_404,
      Component: NotFoundError,
    },
    { path: RouterPath.ERROR_500, Component: GeneralError },
    { path: RouterPath.ERROR_503, Component: MaintenanceError },
    { path: RouterPath.ERROR_401, Component: UnauthorisedError },
  ],
  {
    basename: process.env.NODE_ENV === 'production' ? '' : '/',
  }
)

export default router

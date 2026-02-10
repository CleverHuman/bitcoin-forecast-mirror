import { useState } from 'react'
import { UserAuthForm } from './components/user-auth-form'
import { SignUpForm } from './components/sign-up-form'

export default function SignIn() {
  const [isSignUp, setIsSignUp] = useState(false)

  const toggleForm = () => setIsSignUp(!isSignUp)

  return (
    <div className="flex h-screen items-center justify-center">
      <div className="mx-auto flex w-full max-w-sm flex-col justify-center space-y-4">
        <div className="flex flex-col space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            {isSignUp ? 'Create an Account' : 'Login'}
          </h1>
          <p className="text-sm text-muted-foreground">
            {isSignUp
              ? 'Enter your details below to create your account'
              : 'Enter your email and password below to log into your account'}
          </p>
        </div>

        {isSignUp ? <SignUpForm /> : <UserAuthForm />}

        <p className="text-center text-sm text-muted-foreground">
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button
            onClick={toggleForm}
            className="underline underline-offset-4 hover:text-primary font-bold text-primary"
          >
            {isSignUp ? 'Log in' : 'Sign up'}
          </button>
        </p>

        <p className="text-center text-sm text-muted-foreground">
          By clicking login, you agree to our{' '}
          <a
            href="/terms"
            className="underline underline-offset-4 hover:text-primary"
          >
            Terms of Service
          </a>{' '}
          and{' '}
          <a
            href="/privacy"
            className="underline underline-offset-4 hover:text-primary"
          >
            Privacy Policy
          </a>
          .
        </p>
      </div>
    </div>
  )
}

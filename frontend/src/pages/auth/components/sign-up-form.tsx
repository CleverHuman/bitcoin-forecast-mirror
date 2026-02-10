import { HTMLAttributes, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { IconBrandFacebook, IconBrandGithub } from '@tabler/icons-react'
import { z } from 'zod'
import { useDispatch, useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/custom/button'
import { PasswordInput } from '@/components/custom/password-input'
import { cn } from '@/lib/utils'
import { register, clearErrors } from '@/stores/slices/auth'
import { AppDispatch, RootState } from '@/stores/store'
import { RouterPath } from '@/router/router-path'

interface SignUpFormProps extends HTMLAttributes<HTMLDivElement> {}

const formSchema = z
  .object({
    username: z.string().min(1, { message: 'Username is required' }),
    email: z
      .string()
      .min(1, { message: 'Please enter your email' })
      .email({ message: 'Invalid email address' }),
    password1: z
      .string()
      .min(1, {
        message: 'Please enter your password',
      })
      .min(8, {
        message: 'Password must be at least 8 characters long',
      }),
    password2: z.string(),
  })
  .refine((data) => data.password1 === data.password2, {
    message: "Passwords don't match.",
    path: ['password2'],
  })

type FormData = z.infer<typeof formSchema>

export function SignUpForm({ className, ...props }: SignUpFormProps) {
  const dispatch = useDispatch<AppDispatch>()
  const navigate = useNavigate()
  const { isLoading, errors } = useSelector((state: RootState) => state.auth)

  const form = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: '',
      email: '',
      password1: '',
      password2: '',
    },
  })

  useEffect(() => {
    return () => {
      dispatch(clearErrors())
    }
  }, [dispatch])

  async function onSubmit(data: FormData) {
    try {
      await dispatch(register(data)).unwrap()
      navigate(RouterPath.HOME)
    } catch (error) {
      console.error('Registration failed:', error)
    }
  }

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)}>
          <div className='grid gap-2'>
            <FormField
              control={form.control}
              name='username'
              render={({ field }) => (
                <FormItem className='space-y-1'>
                  <FormLabel>Username</FormLabel>
                  <FormControl>
                    <Input placeholder='johndoe' {...field} />
                  </FormControl>
                  <FormMessage />
                  {errors.username && (
                    <p className="text-red-500 text-sm">{errors.username[0]}</p>
                  )}
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name='email'
              render={({ field }) => (
                <FormItem className='space-y-1'>
                  <FormLabel>Email</FormLabel>
                  <FormControl>
                    <Input placeholder='name@example.com' {...field} />
                  </FormControl>
                  <FormMessage />
                  {errors.email && (
                    <p className="text-red-500 text-sm">{errors.email[0]}</p>
                  )}
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name='password1'
              render={({ field }) => (
                <FormItem className='space-y-1'>
                  <FormLabel>Password</FormLabel>
                  <FormControl>
                    <PasswordInput placeholder='********' {...field} />
                  </FormControl>
                  <FormMessage />
                  {errors.password1 && (
                    <p className="text-red-500 text-sm">{errors.password1[0]}</p>
                  )}
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name='password2'
              render={({ field }) => (
                <FormItem className='space-y-1'>
                  <FormLabel>Confirm Password</FormLabel>
                  <FormControl>
                    <PasswordInput placeholder='********' {...field} />
                  </FormControl>
                  <FormMessage />
                  {errors.password2 && (
                    <p className="text-red-500 text-sm">{errors.password2[0]}</p>
                  )}
                </FormItem>
              )}
            />
            {errors.non_field_errors && (
              <p className="text-red-500 text-sm mt-2">{errors.non_field_errors[0]}</p>
            )}
            
            <Button className='mt-2' loading={isLoading}>
              Create Account
            </Button>

            <div className='relative my-2'>
              <div className='absolute inset-0 flex items-center'>
                <span className='w-full border-t' />
              </div>
              <div className='relative flex justify-center text-xs uppercase'>
                <span className='bg-background px-2 text-muted-foreground'>
                  Or continue with
                </span>
              </div>
            </div>

            <div className='flex items-center gap-2'>
              <Button
                variant='outline'
                className='w-full'
                type='button'
                loading={isLoading}
                leftSection={<IconBrandGithub className='h-4 w-4' />}
              >
                GitHub
              </Button>
              <Button
                variant='outline'
                className='w-full'
                type='button'
                loading={isLoading}
                leftSection={<IconBrandFacebook className='h-4 w-4' />}
              >
                Facebook
              </Button>
            </div>
          </div>
        </form>
      </Form>
    </div>
  )
}
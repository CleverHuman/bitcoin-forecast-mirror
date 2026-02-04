import { useNavigate } from 'react-router-dom'

export default function UnauthenticatedLayout() {
  const navigate = useNavigate()

  return (
    <div className='flex min-h-[calc(100vh-200px)] w-full items-center justify-center'>
    <p className='mx-auto text-center'>
      Please{' '}
      <p
        onClick={() => navigate('/sign-in')}
        className='ml-auto inline-block text-sm underline'
      >
        log in
      </p>{' '}
      to view your dashboards.
      </p>
    </div>
  )
}

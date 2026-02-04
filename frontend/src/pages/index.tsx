import { RouterPath } from '@/router/router-path'
import { Navigate } from 'react-router-dom'

const Home = () => {
  return <Navigate to={RouterPath.FORECASTING} />
}

export default Home

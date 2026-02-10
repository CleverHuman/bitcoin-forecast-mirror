import React, { useEffect } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/auth-context';
import { RouterPath } from './router-path';

const AuthRoute: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate(RouterPath.HOME, { replace: true });
    }
  }, [isAuthenticated, navigate]);

  return !isAuthenticated ? <Outlet /> : null;
};

export default AuthRoute;
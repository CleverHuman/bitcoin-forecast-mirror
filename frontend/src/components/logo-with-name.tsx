import React from 'react';
import Logo from './logo';

interface LogoWithNameProps {
  className?: string;
}

const LogoWithName: React.FC<LogoWithNameProps> = ({ className }) => {
  return (
    <div className={`flex items-center ${className}`}>
      <Logo className="mr-2" />
      <span className="text-lg font-medium">Shadcn Admin</span>
    </div>
  );
};

export default LogoWithName;

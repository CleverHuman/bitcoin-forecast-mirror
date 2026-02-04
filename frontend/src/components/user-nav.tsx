import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/custom/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '@/stores/store';
import { fetchUser, logout } from '@/stores/slices/auth';
import { useAuth } from '@/contexts/auth-context';
import { useUser } from '@/contexts/user-context';

interface IUser {
  avatar_url: string;
  email: string;
  first_name: string;
  last_name: string;
  get_display_name: string;
  id: number;
}

export function UserNav() {
  const navigate = useNavigate();
  const { setSelectedUser } = useUser()
  const [user, setUser] = useState<IUser | null>(null);
  const { isAuthenticated, logout: logoutUser } = useAuth();
  const dispatch = useDispatch<AppDispatch>();
  useEffect(() => {
    async function getUser() {
      try {
        const result = await dispatch(fetchUser());
        setUser(result.payload as any);
        setSelectedUser(result.payload as any)
        
      } catch (error: any) {
        console.error('Login failed:', error);
      }
    }
    getUser();
  }, [dispatch, setSelectedUser]);

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const handleLogout = async (path: string) => {
    try {
      await dispatch(logout());
      logoutUser();
      navigate(path);
    } catch (error: any) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant='ghost' className='relative h-8 w-8 rounded-full'>
          <Avatar className='h-8 w-8'>
            <AvatarImage src='/avatars/01.png' alt='@shadcn' />
            <AvatarFallback>SN</AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className='w-56' align='end' forceMount>
        <DropdownMenuLabel className='font-normal'>
          <div className='flex flex-col space-y-1'>
            <p className='text-sm font-medium leading-none'>
              {isAuthenticated
                ? `${user?.first_name} ${user?.last_name}`
                : 'You are not signed in.'}
            </p>
            <p className='text-xs leading-none text-muted-foreground'>
              {user?.email}
            </p>
          </div>
        </DropdownMenuLabel>
        {isAuthenticated ? (
          <>
            <DropdownMenuGroup>
              <DropdownMenuItem
                onClick={() => handleNavigation('/settings/account')}
              >
                Profile
                <DropdownMenuShortcut>⇧⌘P</DropdownMenuShortcut>
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => handleNavigation('/settings/billing')}
              >
                Billing
                <DropdownMenuShortcut>⌘B</DropdownMenuShortcut>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleNavigation('/settings')}>
                Settings
                <DropdownMenuShortcut>⌘S</DropdownMenuShortcut>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleNavigation('/new-team')}>
                New Team
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => handleLogout('/sign-in')}>
              Log out
              <DropdownMenuShortcut>⇧⌘Q</DropdownMenuShortcut>
            </DropdownMenuItem>
          </>
        ) : (
          <DropdownMenuItem onClick={() => handleNavigation('/sign-in')}>
            Log in
            <DropdownMenuShortcut>⇧⌘Q</DropdownMenuShortcut>
          </DropdownMenuItem>
        )}
        <DropdownMenuSeparator />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

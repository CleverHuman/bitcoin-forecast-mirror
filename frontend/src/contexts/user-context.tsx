import React, { createContext, useContext, useState } from 'react'

type User = {
  id: number;
  first_name: string;
  last_name: string;
  email: string,
  avatar_url: string;
  get_display_name: string;
}

interface UserContextType {
  selectedUser: User | null
  setSelectedUser: React.Dispatch<React.SetStateAction<User | null>>
}

const UserContext = createContext<UserContextType | undefined>(undefined)

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [selectedUser, setSelectedUser] = useState<User | null>(null)

  return (
    <UserContext.Provider value={{ selectedUser, setSelectedUser }}>
      {children}
    </UserContext.Provider>
  )
}

export const useUser = (): UserContextType => {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error('useUser must be used within a UserProvider')
  }
  return context
}

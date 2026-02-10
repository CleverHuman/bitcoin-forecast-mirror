import React, { createContext, useContext, useState } from 'react'

type Team = {
  id?: string
  label: string
  value: string
  slug?: string
}

interface TeamContextType {
  selectedTeam: Team | null
  setSelectedTeam: React.Dispatch<React.SetStateAction<Team | null>>
}

const TeamContext = createContext<TeamContextType | undefined>(undefined)

export const TeamProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null)

  return (
    <TeamContext.Provider value={{ selectedTeam, setSelectedTeam }}>
      {children}
    </TeamContext.Provider>
  )
}

export const useTeam = (): TeamContextType => {
  const context = useContext(TeamContext)
  if (!context) {
    throw new Error('useTeam must be used within a TeamProvider')
  }
  return context
}

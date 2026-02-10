import { Link } from 'react-router-dom'
import { IconChevronDown, IconSearch } from '@tabler/icons-react'
import { Button, buttonVariants } from './custom/button'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './ui/tooltip'
import { cn } from '@/lib/utils'
import useCheckActiveNav from '@/hooks/use-check-active-nav'
import { SideLink } from '@/data/sidelinks'
import { useSearch } from '@/contexts/search-context'
import { RouterPath } from '@/router/router-path'

interface NavProps extends React.HTMLAttributes<HTMLDivElement> {
  isCollapsed: boolean
  links: SideLink[]
  closeNav: () => void
}

export default function Nav({
  links,
  isCollapsed,
  className,
  closeNav,
}: NavProps) {

  const { openSearch } = useSearch()


  const handleLinkClick = (link: SideLink) => {
    if (link.href === '/search') {
      openSearch()
    } else {
      closeNav()
    }
  }


  const renderLink = ({ sub, ...rest }: SideLink) => {
    const key = `${rest.title}-${rest.href}`
    
    const commonProps = {
      ...rest,
      closeNav: (link: SideLink) => handleLinkClick(link),
    }

    if (rest.href === RouterPath.SEARCH && !isCollapsed) 
      return <SearchBar key={key} {...commonProps} />

    if (isCollapsed && sub)
      return <NavLinkIconDropdown key={key} {...commonProps} sub={sub} />

    if (isCollapsed) 
      return <NavLinkIcon key={key} {...commonProps} />

    if (sub) 
      return <NavLinkDropdown key={key} {...commonProps} sub={sub} />

    return <NavLink key={key} {...commonProps} />
  }
  return (
    <div
      data-collapsed={isCollapsed}
      className={cn(
        'group bg-background py-2 transition-[max-height,padding] duration-500 data-[collapsed=true]:py-2 border-none',
        className
      )}
    >
      <TooltipProvider delayDuration={0}>
        <nav className='grid gap-1 group-[[data-collapsed=true]]:justify-center group-[[data-collapsed=true]]:px-2'>
          {links.map(renderLink)}
        </nav>
      </TooltipProvider>
    </div>
  )
}

interface SearchBarProps extends SideLink {
  closeNav: (link: SideLink) => void
}

function SearchBar({ title, icon, closeNav }: SearchBarProps) {
  const { openSearch } = useSearch()

  const handleClick = () => {
    openSearch()
    closeNav({ title, icon, href: '/search' })
  }

  return (
   <div className='mx-2'>
     <button
      className={cn(
        buttonVariants({
          variant: 'outline',
          size: 'sm',
        }),
        'h-9 justify-between text-wrap rounded-md px-3 w-full bg-background/95 text-muted-foreground'
      )}
      onClick={handleClick}
    >
      <div className="flex items-center">
        <IconSearch size={16} className="mr-2" />
        <span className="text-[0.625rem] sm:text-xs">Search...</span>
      </div>
      <div className="text-[0.625rem] bg-muted text-muted-foreground px-1.5 py-0.5 rounded hidden sm:block">
        ⌘ K
      </div>
    </button>
   </div>
  )
}


interface NavLinkProps extends SideLink {
  subLink?: boolean
  closeNav: (link: SideLink) => void
}

function NavLink(navlink: NavLinkProps) {
  const { checkActiveNav } = useCheckActiveNav()
  return (
    <Link
      to={navlink.href}
      onClick={() => navlink.closeNav(navlink)}
      className={cn(
        buttonVariants({
          variant: checkActiveNav(navlink.href) ? 'secondary' : 'ghost',
          size: 'sm',
        }),
        'h-12 justify-start text-wrap rounded-none px-6',
        navlink.subLink && 'h-10 w-full px-2'
      )}
      aria-current={checkActiveNav(navlink.href) ? 'page' : undefined}
    >
      <div className='mr-2'>{navlink.icon}</div>
      {navlink.title}
      {navlink.label && (
        <div className='ml-2 rounded-lg bg-primary px-1 text-[0.625rem] text-primary-foreground'>
          {navlink.label}
        </div>
      )}
    </Link>
  )
}

function NavLinkDropdown({ title, icon, label, sub, closeNav }: NavLinkProps) {
  const { checkActiveNav } = useCheckActiveNav()

  /* Open collapsible by default
   * if one of child element is active */
  const isChildActive = !!sub?.find((s) => checkActiveNav(s.href))

  return (
    <Collapsible defaultOpen={isChildActive}>
      <CollapsibleTrigger
        className={cn(
          buttonVariants({ variant: 'ghost', size: 'sm' }),
          'group h-12 w-full justify-start rounded-none px-6'
        )}
      >
        <div className='mr-2'>{icon}</div>
        {title}
        {label && (
          <div className='ml-2 rounded-lg bg-primary px-1 text-[0.625rem] text-primary-foreground'>
            {label}
          </div>
        )}
        <span
          className={cn(
            'ml-auto transition-all group-data-[state="open"]:-rotate-180'
          )}
        >
          <IconChevronDown stroke={1} />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent className='collapsibleDropdown' asChild>
        <ul>
          {sub!.map((sublink) => (
            <li key={sublink.title} className='my-1 ml-8'>
              <NavLink {...sublink} subLink closeNav={closeNav} />
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  )
}

function NavLinkIcon({ title, icon, label, href, closeNav }: NavLinkProps) {
  const { checkActiveNav } = useCheckActiveNav()
  const { openSearch } = useSearch()

  const handleClick = (e: React.MouseEvent) => {
    if (href === RouterPath.SEARCH) {
      e.preventDefault()
      openSearch()
      closeNav({ title, icon, href, label })
    }
  }

  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild>
        <Link
          to={href}
          onClick={handleClick}
          className={cn(
            buttonVariants({
              variant: checkActiveNav(href) ? 'secondary' : 'ghost',
              size: 'icon',
            }),
            'h-12 w-12'
          )}
        >
          {icon}
          <span className='sr-only'>{title}</span>
        </Link>
      </TooltipTrigger>
      <TooltipContent side='right' className='flex items-center gap-4'>
        {title}
        {label && (
          <span className='ml-auto text-muted-foreground'>{label}</span>
        )}
      </TooltipContent>
    </Tooltip>
  )
}

function NavLinkIconDropdown({ title, icon, label, sub }: NavLinkProps) {
  const { checkActiveNav } = useCheckActiveNav()

  /* Open collapsible by default
   * if one of child element is active */
  const isChildActive = !!sub?.find((s) => checkActiveNav(s.href))

  return (
    <DropdownMenu>
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button
              variant={isChildActive ? 'secondary' : 'ghost'}
              size='icon'
              className='h-12 w-12'
            >
              {icon}
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side='right' className='flex items-center gap-4'>
          {title}{' '}
          {label && (
            <span className='ml-auto text-muted-foreground'>{label}</span>
          )}
          <IconChevronDown
            size={18}
            className='-rotate-90 text-muted-foreground'
          />
        </TooltipContent>
      </Tooltip>
      <DropdownMenuContent side='right' align='start' sideOffset={4}>
        <DropdownMenuLabel>
          {title} {label ? `(${label})` : ''}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        {sub!.map(({ title, icon, label, href }) => (
          <DropdownMenuItem key={`${title}-${href}`} asChild>
            <Link
              to={href}
              className={`${checkActiveNav(href) ? 'bg-secondary' : ''}`}
            >
              {icon} <span className='ml-2 max-w-52 text-wrap'>{title}</span>
              {label && <span className='ml-auto text-xs'>{label}</span>}
            </Link>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

import {lazy,Suspense,useEffect,useRef,useState} from 'react';
import {NavLink,Route,Routes,useLocation} from 'react-router-dom';
import {useMutation,useQuery} from '@tanstack/react-query';
import {AnimatePresence,motion,useReducedMotion} from 'framer-motion';
import {Activity,ArrowUpRight,ArrowRight,Bolt,ChartNoAxesCombined,ChevronRight,Command as CommandIcon,Database,FlaskConical,Layers3,LayoutGrid,Menu,Radio,Settings2,Wind,X} from 'lucide-react';
import {api,errorText} from './api';
import {useDesk} from './store';
import {ForecastContext} from './context';
import {Notice} from './components/ui';
import type {Case} from './types';
const Command=lazy(()=>import('./pages/Command'));
const Analytical=lazy(()=>import('./pages/Analytical'));
const nav=[['/','Command Center',LayoutGrid],['/demand','Grid Demand Intelligence',Activity],['/renewables','Renewable Energy Monitoring',Wind],['/forecast-lab','AI Forecasting Lab',FlaskConical],['/grid-requirement','Grid Requirement Analysis',Layers3],['/models','Model Intelligence',ChartNoAxesCombined],['/history','Historical Data Explorer',Database],['/operations','System Operations',Settings2]] as const;
export default function App(){const location=useLocation();const reduced=useReducedMotion();const[mobile,setMobile]=useState(false);const[hidden,setHidden]=useState(document.hidden);
 const draft=useDesk(s=>s.draftOrigin);const setDraft=useDesk(s=>s.setDraft);const result=useDesk(s=>s.forecast);const setForecast=useDesk(s=>s.setForecast);
 const status=useQuery({queryKey:['status'],queryFn:api.status,refetchInterval:15000,refetchOnWindowFocus:false});
 const mutation=useMutation({mutationFn:({origin,kind}:{origin:string;kind:Case})=>api.forecast(origin,kind),onSuccess:setForecast});
 const run=(origin=draft,kind:Case='normal')=>mutation.mutate({origin,kind});
 const bootstrap=useRef(false);useEffect(()=>{if(!bootstrap.current){bootstrap.current=true;if(!result)run()}},[]);
 useEffect(()=>{const listener=()=>setHidden(document.hidden);document.addEventListener('visibilitychange',listener);return()=>document.removeEventListener('visibilitychange',listener)},[]);
 useEffect(()=>setMobile(false),[location.pathname]);
 useEffect(()=>{const close=(event:KeyboardEvent)=>{if(event.key==='Escape'){setMobile(false);document.querySelector<HTMLButtonElement>('[aria-label="Toggle navigation"]')?.focus()}};document.addEventListener('keydown',close);return()=>document.removeEventListener('keydown',close)},[]);
 const changed=result&&draft.replace('T',' ').slice(0,16)!==result.forecast.origin.slice(0,16);
 return <ForecastContext.Provider value={{pending:mutation.isPending,error:mutation.error?errorText(mutation.error):null,run}}><div className={`app-shell ${hidden?'page-hidden':''}`}>
 <aside className={`nav-rail ${mobile?'open':''}`}><NavLink to="/" className="brand"><span className="brand-symbol"><Bolt size={23}/></span><div>GRIDCAST<span>AI</span><small>ENERGY INTELLIGENCE SYSTEM</small></div></NavLink><div className="rail-label">WORKSPACE / INDIA</div><nav aria-label="Main navigation">{nav.map(([path,label,Icon],i)=><NavLink key={path} to={path} end={path==='/'} className={({isActive})=>isActive?'nav-item active':'nav-item'}><Icon size={17}/><span>{label}</span><small>{String(i+1).padStart(2,'0')}</small></NavLink>)}</nav><div className="rail-bottom"><div className="protocol"><span className="status-dot"/> LOCAL RESEARCH ENVIRONMENT</div><p>National aggregate models<br/>Historical source observations</p><span className="rail-version">GRID_CURVE_V1 <ArrowUpRight size={12}/></span></div></aside>
 {mobile&&<button className="mobile-backdrop" aria-label="Close navigation" onClick={()=>setMobile(false)}/>}
 <main className="workspace"><header className="topbar"><button className="icon-button mobile-nav" aria-label="Toggle navigation" onClick={()=>setMobile(!mobile)}>{mobile?<X size={18}/>:<Menu size={18}/>}</button><div className="breadcrumb"><CommandIcon size={14}/><span>INTELLIGENCE PLATFORM</span><ChevronRight size={12}/><b>{nav.find(([p])=>p===location.pathname)?.[1].toUpperCase()??'PAGE NOT FOUND'}</b></div><div className="top-status"><span className={`status-dot ${status.isError?'error':''}`}/><span>{status.isError?'API DISCONNECTED':'LOCAL API'}</span><span className="chip">RESEARCH / RETROSPECTIVE</span></div></header>
 <section className="issue-bar" aria-label="Forecast issue controls"><div className="issue-label"><Radio size={16}/><span>ISSUE TIME<small>SOURCE CLOCK</small></span></div><label className="sr-only" htmlFor="origin">Forecast origin</label><input id="origin" type="datetime-local" step="3600" value={draft} onChange={e=>setDraft(e.target.value)}/><button className="primary-button" onClick={()=>run()} disabled={mutation.isPending||!draft}>{mutation.isPending?<><span className="spinner"/>Running saved models</>:<>Run forecast <ArrowRight size={14}/></>}</button><div className="issue-context"><span>{changed?'DRAFT CHANGED · RUN TO APPLY':result?`DISPLAYED ISSUE ${result.forecast.origin.slice(0,16)}`:'LOADING SAVED FORECAST METHODS'}</span><small>{result?`24 horizons · ${result.forecast.model_id}`:'No training · no battery solver'}</small></div></section>
 {result?.synthetic&&<div className="shell-notice"><Notice critical>DEMONSTRATION INPUT · Synthetic blocked wind. Source measurements are unchanged; this label follows the result across every page.</Notice></div>}
 {mutation.error&&<div className="shell-notice"><Notice critical>Forecast unavailable: {errorText(mutation.error)}. {result?'The last successful result is retained.':'Historical views remain independently available.'}</Notice></div>}
 <AnimatePresence mode="wait"><motion.div key={location.pathname} initial={reduced?false:{opacity:0,y:7}} animate={{opacity:1,y:0}} exit={reduced?{opacity:1}:{opacity:0,y:-4}} transition={{duration:.18}}><Suspense fallback={<div className="route-loading"><span className="spinner"/>Loading analytical view</div>}><Routes location={location}><Route path="/" element={<Command/>}/>{['demand','renewables','forecast-lab','grid-requirement','models','history','operations'].map(p=><Route key={p} path={'/'+p} element={<Analytical page={p}/>}/>)}<Route path="*" element={<div className="page"><h1>Page not found</h1><NavLink to="/">Return to Command Center</NavLink></div>}/></Routes></Suspense></motion.div></AnimatePresence>
 <footer className="app-footer"><span>GRIDCASTAI / NATIONAL ELECTRICITY INTELLIGENCE</span><span>LOCAL DEMONSTRATION · OPTIMIZATION DEFERRED</span></footer></main></div></ForecastContext.Provider>;
}

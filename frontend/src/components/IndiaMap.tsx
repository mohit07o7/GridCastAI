import {useEffect,useRef,useState} from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
import {LocateFixed} from 'lucide-react';
export default function IndiaMap(){const container=useRef<HTMLDivElement>(null);const map=useRef<maplibregl.Map|null>(null);const[failed,setFailed]=useState(false);const[selected,setSelected]=useState(true);
 useEffect(()=>{if(!container.current)return;let disposed=false;let instance:maplibregl.Map|undefined;let observer:ResizeObserver|undefined;
 try{
  maplibregl.setWorkerUrl(workerUrl);
  if(new URLSearchParams(location.search).has('mapFallback'))throw Error('Requested WebGL fallback');
  instance=new maplibregl.Map({container:container.current,center:[80,22.5],zoom:3.25,pitch:0,bearing:0,attributionControl:false,style:{version:8,sources:{},layers:[{id:'background',type:'background',paint:{'background-color':'#07101e'}}]}});
  map.current=instance;instance.addControl(new maplibregl.NavigationControl({showCompass:false}),'bottom-right');
  instance.on('load',()=>{if(disposed||!instance)return;instance.addSource('india',{type:'geojson',data:'/india.geojson'});
   instance.addLayer({id:'india-fill',type:'fill',source:'india',paint:{'fill-color':'#00D9FF','fill-opacity':.065}});
   instance.addLayer({id:'india-glow',type:'line',source:'india',paint:{'line-color':'#00D9FF','line-width':7,'line-opacity':.09,'line-blur':5}});
   instance.addLayer({id:'india-outline',type:'line',source:'india',paint:{'line-color':'#00D9FF','line-width':1.2,'line-opacity':.8}});
   instance.fitBounds([[67,6],[98,37]],{padding:28,duration:0});instance.on('click','india-fill',()=>setSelected(true));
  });
  instance.on('error',()=>{if(!disposed)setFailed(true)});
  observer=new ResizeObserver(()=>instance?.resize());observer.observe(container.current);
 }catch{setFailed(true)}
 return()=>{disposed=true;observer?.disconnect();instance?.remove();map.current=null};},[]);
 return <div className="map-stage"><div className="map-grid"/><div ref={container} className={`map-canvas ${failed?'invisible':''}`}/>{failed&&<div className="static-map" role="img" aria-label="Static India geographic outline"><svg viewBox="0 0 400 450"><IndiaOutline/></svg><span>STATIC GEOGRAPHIC REFERENCE · WEBGL UNAVAILABLE</span></div>}
 <div className="map-coordinate">NATIONAL COVERAGE / IN<br/><span>06°–37° N · 67°–98° E</span></div>
 <button className="map-reset" aria-label="Reset India map" onClick={()=>map.current?.fitBounds([[67,6],[98,37]],{padding:28,duration:300})}><LocateFixed size={15}/> Recenter</button>
 <button className="coverage-callout" aria-pressed={selected} onClick={()=>setSelected(!selected)}><span className="status-dot"/><div><small>MODEL COVERAGE</small><strong>INDIA / NATIONAL</strong><span>{selected?'Aggregate forecast coverage selected':'Select national coverage'}</span></div></button>
 <div className="map-attribution">Geographic reference · national aggregate coverage<br/><a href="https://www.naturalearthdata.com/about/terms-of-use/" target="_blank" rel="noreferrer">Made with Natural Earth</a> · No regional telemetry</div></div>;
}
function IndiaOutline(){const[path,setPath]=useState('');useEffect(()=>{fetch('/india.geojson').then(r=>r.json()).then((geo:{features:{geometry:{type:string;coordinates:number[][][]|number[][][][]}}[]})=>{const g=geo.features[0].geometry;const polygons=g.type==='MultiPolygon'?g.coordinates as number[][][][]:[g.coordinates as number[][][]];setPath(polygons.map(p=>p.map(r=>r.map(([x,y],i)=>`${i?'L':'M'}${(x-67)*12},${(38-y)*12}`).join(' ')+'Z').join(' ')).join(' '))}).catch(()=>setPath(''));},[]);return <path d={path} fill="#00d9ff15" stroke="#00D9FF" strokeWidth="1.2"/>}

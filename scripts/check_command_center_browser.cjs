const fs=require('fs');const path=require('path');const assert=require('node:assert/strict');
const {chromium}=require(process.env.GRIDCAST_PLAYWRIGHT || '/tmp/gridcast-dashboard-browser/node_modules/playwright');
const root=path.resolve(__dirname,'..');const base=path.join(root,'reports/demos/command_center_v1');fs.mkdirSync(base,{recursive:true});const output=fs.mkdtempSync(path.join(base,'browser-'));
const url='http://127.0.0.1:8502';
const pause=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.GRIDCAST_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-swiftshader']});
 const context=await browser.newContext({viewport:{width:1600,height:1080},acceptDownloads:true});const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));const checks={};
 const shot=async name=>{await pause(500);await page.screenshot({path:path.join(output,name+'.png'),fullPage:true})};
 try{
  await page.goto(url);await page.getByText('COMPLETE FORECAST CURVE',{exact:true}).waitFor({timeout:90000});
  await page.getByRole('button',{name:'Zoom in',exact:true}).waitFor();await pause(600);
  await page.getByRole('button',{name:'Zoom in',exact:true}).click();await page.getByRole('button',{name:'Reset India map'}).click();
  const map=page.locator('.maplibregl-canvas');const box=await map.boundingBox();assert(box&&box.height>300);
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+45,box.y+box.height/2+10,{steps:6});await page.mouse.up();await page.getByRole('button',{name:'Reset India map'}).click();
  await page.getByRole('button',{name:/INDIA \/ NATIONAL/}).click();
  checks.real_map_zoom_pan_reset_selection=true;await shot('01-command-center');
  await page.locator('.echart').first().scrollIntoViewIfNeeded();await pause(200);
  await page.locator('.echart').first().hover({position:{x:220,y:110}});await pause(250);
  const tooltips=await page.locator('.echart').allInnerTexts();
  assert(tooltips.length>=2&&tooltips[0].split('\n')[0].startsWith('2025-')&&tooltips[0].split('\n')[0]===tooltips[1].split('\n')[0]);
  await page.mouse.move(0,0);checks.synchronized_target_time_tooltips=true;
  await page.getByRole('slider',{name:'Highlighted forecast horizon'}).focus();await page.keyboard.press('Home');
  await page.getByText('Target horizon +1h',{exact:true}).waitFor();checks.highlighted_target_control=true;
  for(const [route,title] of [['demand','GRID DEMAND INTELLIGENCE.'],['renewables','RENEWABLE ENERGY MONITORING.'],['forecast-lab','AI FORECASTING LAB.'],['grid-requirement','GRID REQUIREMENT ANALYSIS.'],['models','MODEL INTELLIGENCE.'],['history','HISTORICAL DATA EXPLORER.'],['operations','SYSTEM OPERATIONS.']]){
   await page.locator(`nav a[href="/${route}"]`).click();await page.getByRole('heading',{level:1,name:title,exact:true}).waitFor();await pause(700);assert(await page.locator('canvas').count()>0||route==='operations');await shot('page-'+route);
  }
  checks.all_eight_routes_render_real_data=true;
  await page.goBack();await page.getByRole('heading',{level:1,name:'HISTORICAL DATA EXPLORER.',exact:true}).waitFor();await page.goForward();await page.getByRole('heading',{level:1,name:'SYSTEM OPERATIONS.',exact:true}).waitFor();checks.browser_back_forward=true;
  await page.locator('nav a[href="/models"]').click();
  await page.getByLabel('Evaluation quantity',{exact:true}).selectOption('solar');await page.getByLabel('Evaluation horizon',{exact:true}).selectOption('24');await page.getByLabel('Scoring cohort',{exact:true}).selectOption('selected_only');await page.getByLabel('Evaluation month',{exact:true}).selectOption('2025-04');
  const expected=await (await context.request.get(url+'/api/v1/metrics?quantity=solar&horizon=24&cohort=selected_only&month=2025-04')).json();
  const displayed=page.locator('.metric').filter({has:page.getByText('Forecast MAE',{exact:true})}).locator('.metric-number');
  await page.waitForFunction(expectedValue=>{const metric=[...document.querySelectorAll('.metric')].find(m=>m.querySelector('.eyebrow')?.textContent==='Forecast MAE');return metric?.textContent?.includes(expectedValue)},expected.saved.forecast_mae_mw.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2}));
  await page.getByText('Derived MAPE uses',{exact:false}).waitFor();checks.metric_filter_matches_saved_api_and_mape_label=true;
  await page.getByLabel('Importance component',{exact:true}).selectOption('wind');await pause(400);await page.getByText('Candidate: wind_h24',{exact:false}).waitFor();checks.native_selected_tree_importance=true;
  await page.locator('nav a[href="/forecast-lab"]').click();
  const [normalResponse]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/api/v1/forecast')&&r.request().method()==='POST'),page.getByRole('button',{name:'Normal forecast',exact:true}).click()]);const normal=await normalResponse.json();
  let waiting=page.waitForEvent('download');await page.getByRole('button',{name:'JSON',exact:true}).click();let dl=await waiting;await dl.saveAs(path.join(output,'normal.json'));assert.deepEqual(JSON.parse(fs.readFileSync(path.join(output,'normal.json'))),normal.forecast);
  waiting=page.waitForEvent('download');await page.getByRole('button',{name:'CSV',exact:true}).click();dl=await waiting;await dl.saveAs(path.join(output,'normal.csv'));const csv=fs.readFileSync(path.join(output,'normal.csv'),'utf8');assert.equal(csv.trim().split('\n').length,25);checks.forecast_downloads_match_api=true;
  await page.locator('#origin').fill('2025-06-23T23:00');await page.getByText('DRAFT CHANGED · RUN TO APPLY',{exact:true}).waitFor();assert(await page.getByText('DISPLAYED ORIGIN 2025-06-24 23:00:00',{exact:true}).count());checks.changed_control_retains_issued_forecast=true;
  await page.route('**/api/v1/forecast',async route=>{await pause(1400);await route.continue()});
  await page.getByRole('button',{name:'Run forecast',exact:true}).click();await page.getByRole('button',{name:'Running saved models',exact:true}).waitFor();checks.genuine_pending_state=true;
  await page.getByRole('button',{name:'Run forecast',exact:true}).waitFor({timeout:30000});await page.unroute('**/api/v1/forecast');
  await page.getByRole('button',{name:'Historical fallback',exact:true}).click();await page.getByText('DISPLAYED ORIGIN 2025-01-09 15:00:00',{exact:true}).waitFor({timeout:30000});
  await page.getByRole('checkbox',{name:'Actual overlay',exact:true}).check();await page.getByText('24 reported actual hours. This overlay is attached after prediction.',{exact:true}).waitFor();await shot('09-historical-fallback');checks.labelled_fallback_and_separate_actual_overlay=true;
  await page.getByRole('button',{name:'Synthetic blocked wind',exact:true}).click();await page.getByText('DEMONSTRATION INPUT · Synthetic blocked wind.',{exact:false}).waitFor({timeout:30000});
  waiting=page.waitForEvent('download');await page.getByRole('button',{name:'JSON',exact:true}).click();dl=await waiting;await dl.saveAs(path.join(output,'blocked.json'));const blocked=JSON.parse(fs.readFileSync(path.join(output,'blocked.json')));assert(blocked.points.every(p=>p.components.wind.forecast_mw===null&&p.net_demand_mw===null));await shot('10-synthetic-blocked');checks.blocked_components_are_null=true;
  await page.route('**/api/v1/forecast',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Browser-test API outage fixture'})}));
  await page.getByRole('button',{name:'Run forecast',exact:true}).click();await page.getByText('The last successful result is retained.',{exact:false}).waitFor();assert(await page.getByText('DEMONSTRATION INPUT · Synthetic blocked wind.',{exact:false}).count());await shot('11-api-outage-fixture');await page.unroute('**/api/v1/forecast');checks.api_outage_retains_last_result=true;
  await page.locator('nav a[href="/history"]').click();await page.getByLabel('Historical start date',{exact:true}).fill('2025-04-04');await page.getByLabel('Historical end date',{exact:true}).fill('2025-04-04');await page.getByRole('button',{name:'Apply range',exact:true}).click();
  await page.waitForFunction(()=>[...document.querySelectorAll('.metric')].some(m=>m.querySelector('.eyebrow')?.textContent==='Selected source hours'&&m.querySelector('.metric-number')?.textContent==='24'));
  waiting=page.waitForEvent('download');await page.getByRole('link',{name:'Download full-resolution CSV',exact:false}).click();dl=await waiting;await dl.saveAs(path.join(output,'history.csv'));assert.equal(fs.readFileSync(path.join(output,'history.csv'),'utf8').trim().split('\n').length,25);checks.history_full_resolution_download=true;
  await page.locator('nav a[href="/operations"]').click();await page.getByLabel('Evaluation month',{exact:true}).selectOption('2025-04');await page.getByText('Battery optimization remains Deferred.',{exact:false}).waitFor();checks.optimization_deferred=true;
  await page.route('**/api/v1/models?**',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Browser-test model outage; artifact files unchanged'})}));
  await page.locator('nav a[href="/models"]').click();await page.getByLabel('Importance component',{exact:true}).selectOption('solar');await page.getByText('Browser-test model outage; artifact files unchanged',{exact:false}).waitFor();assert(await page.getByText('METRIC REGISTER',{exact:true}).count());await shot('12-model-outage-fixture');await page.unroute('**/api/v1/models?**');checks.model_outage_preserves_accuracy=true;
  await page.setViewportSize({width:1050,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await shot('13-narrow');
  await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'Toggle navigation',exact:true}).click();await page.locator('nav a[href="/"]').click();assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await shot('14-mobile');
  await page.getByRole('button',{name:'Toggle navigation',exact:true}).click();await page.keyboard.press('Escape');assert(await page.locator('aside').evaluate(e=>getComputedStyle(e).visibility==='hidden'));checks.responsive_keyboard_navigation=true;
  const fallback=await context.newPage();await fallback.goto(url+'/?mapFallback=1');await fallback.getByText('STATIC GEOGRAPHIC REFERENCE · WEBGL UNAVAILABLE',{exact:true}).waitFor({timeout:30000});await fallback.screenshot({path:path.join(output,'15-static-map-fallback.png'),fullPage:true});await fallback.close();checks.webgl_fallback=true;
  const reducedContext=await browser.newContext({viewport:{width:1280,height:900},reducedMotion:'reduce'});const reduced=await reducedContext.newPage();await reduced.goto(url);await reduced.getByText('COMPLETE FORECAST CURVE',{exact:true}).waitFor({timeout:60000});assert(await reduced.evaluate(()=>matchMedia('(prefers-reduced-motion: reduce)').matches));await reduced.screenshot({path:path.join(output,'16-reduced-motion.png'),fullPage:true});await reducedContext.close();checks.reduced_motion=true;
  assert.deepEqual(errors,[]);checks.no_browser_exceptions=true;
  fs.writeFileSync(path.join(output,'browser_verification.json'),JSON.stringify({status:'passed',checks,errors,browser_version:browser.version()},null,2));console.log(JSON.stringify({status:'passed',output,checks},null,2));
 }catch(error){await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});fs.writeFileSync(path.join(output,'failure.txt'),String(error));console.error(String(error));process.exitCode=1}
 finally{await browser.close()}
})();

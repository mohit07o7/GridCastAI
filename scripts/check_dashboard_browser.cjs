/* Optional real-Chrome smoke test. Install the pinned driver outside the repo. */
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {chromium} = require(process.env.GRIDCAST_PLAYWRIGHT || '/tmp/gridcast-dashboard-browser/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = fs.mkdtempSync(path.join(root, 'reports/demos/dashboard_v1/browser-'));
const expected = JSON.parse(execFileSync(path.join(root,'.venv/bin/python'),['-c',
  "import sys,json;sys.path.insert(0,'src');from dashboard.service import load_reports,metric_record,audit_slice;r=load_reports();_,c,_=audit_slice(r,1,'2025-04');print(json.dumps({'h1':f'{metric_record(r,horizon=1).forecast_mae_mw:,.2f} MW','h24':f'{metric_record(r).forecast_mae_mw:,.2f} MW','april_h1_excluded':c['excluded_target_records']}))"],
  {cwd:root,encoding:'utf8',env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}}));
const runs=fs.readdirSync(path.join(root,'reports/demos/dashboard_v1')).filter(n=>n.startsWith('run-'))
  .map(n=>path.join(root,'reports/demos/dashboard_v1',n)).sort((a,b)=>fs.statSync(b).mtimeMs-fs.statSync(a).mtimeMs);
const golden = process.env.GRIDCAST_DEMO_RUN || runs[0];
if (!golden) throw Error('Run scripts/run_dashboard_demo.py before the browser check');

(async () => {
  const browser = await chromium.launch({executablePath:process.env.GRIDCAST_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
  const page = await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const checks={};
  try {
    await page.goto('http://127.0.0.1:8501',{waitUntil:'domcontentloaded'});
    await page.getByText('Complete curve · all selected methods available',{exact:true}).waitFor({timeout:90000});
    await page.locator('.js-plotly-plot').first().waitFor({timeout:30000});
    const first = await page.locator('.js-plotly-plot').first().evaluate(e=>e.data.map(t=>({name:t.name,targets:t.x.length,connectgaps:t.connectgaps})));
    assert.equal(first.length,3);assert(first.every(x=>x.targets===24 && x.connectgaps===false));
    checks.aligned_component_charts=true;
    await page.screenshot({path:path.join(output,'01-normal.png'),fullPage:true});
    for (const [label,name] of [['Download 24-hour CSV','normal.csv'],['Download forecast JSON','normal.json']]) {
      const waiting=page.waitForEvent('download');
      await page.getByRole('button',{name:label,exact:true}).click();
      const download=await waiting;
      const dest=path.join(output,name);await download.saveAs(dest);
      assert(fs.readFileSync(dest).equals(fs.readFileSync(path.join(golden,name))));
    }
    await page.screenshot({path:path.join(output,'01b-normal-details.png'),fullPage:true});
    checks.downloads_match_service=true;
    await page.getByRole('tab',{name:'Historical accuracy',exact:true}).click();
    let metric=page.getByTestId('stMetric').filter({hasText:'Forecast MAE'}).getByTestId('stMetricValue');
    await metric.getByText(expected.h24,{exact:true}).waitFor();
    const horizon=page.getByTestId('stSelectbox').filter({hasText:'Forecast horizon'}).getByRole('combobox');
    await horizon.click();await page.getByRole('option',{name:'1 hours',exact:true}).click();
    await metric.getByText(expected.h1,{exact:true}).waitFor();
    await page.screenshot({path:path.join(output,'02-accuracy.png'),fullPage:true});
    checks.historical_metric_filter=true;
    const month=page.getByTestId('stSelectbox').filter({hasText:'Target month'}).getByRole('combobox');
    await month.click();await page.getByRole('option',{name:'2025-04',exact:true}).click();
    await page.getByRole('tab',{name:'Quality and exclusions',exact:true}).click();
    await page.getByText('After observation: retrospective target audit',{exact:true}).waitFor();
    const excluded=page.getByTestId('stMetric').filter({hasText:'Excluded target records'}).getByTestId('stMetricValue');
    await excluded.getByText(String(expected.april_h1_excluded),{exact:true}).waitFor();
    await page.screenshot({path:path.join(output,'03-quality.png'),fullPage:true});
    checks.quality_audit_separate=true;
    await page.getByRole('tab',{name:'Optimization status',exact:true}).click();
    await page.getByRole('heading',{name:'Optimization: Deferred',exact:true}).waitFor();
    await page.screenshot({path:path.join(output,'04-deferred.png'),fullPage:true});
    checks.optimization_deferred=true;
    await page.getByRole('button',{name:'Historical fallback',exact:true}).click();
    await page.getByRole('tab',{name:'24-hour forecast',exact:true}).click();
    await page.getByText('Complete curve · clearly labelled fallback methods are included',{exact:true}).waitFor({timeout:30000});
    await page.getByText('Overlay reported historical actuals',{exact:true}).click();
    assert(await page.getByRole('checkbox',{name:'Overlay reported historical actuals',exact:true}).isChecked());
    await page.getByText('Reported actuals available for 24/24 target hours.',{exact:true}).waitFor();
    await page.screenshot({path:path.join(output,'05-fallback.png'),fullPage:true});
    checks.fallback_and_actual_overlay=true;
    await page.getByRole('button',{name:'Synthetic blocked wind',exact:true}).click();
    await page.getByText('Incomplete curve · blocked components remain missing; gaps are not filled with zero',{exact:true}).waitFor({timeout:30000});
    await page.screenshot({path:path.join(output,'06-blocked.png'),fullPage:true});
    const blocked = await page.locator('.js-plotly-plot').first().evaluate(e=>e.data.find(t=>t.name==='Wind').y);
    // The CSV check below verifies missing values independently of plotly's binary encoding.
    const waiting=page.waitForEvent('download');await page.getByRole('button',{name:'Download 24-hour CSV',exact:true}).click();
    const dl=await waiting;await dl.saveAs(path.join(output,'blocked.csv'));
    assert(fs.readFileSync(path.join(output,'blocked.csv')).equals(fs.readFileSync(path.join(golden,'blocked.csv'))));
    checks.blocked_component_and_download=true;
    await page.setViewportSize({width:1050,height:850});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    await page.screenshot({path:path.join(output,'07-narrow.png'),fullPage:true});
    checks.narrow_layout_no_page_overflow=true;
    assert.deepEqual(errors,[]);checks.no_browser_exceptions=true;
    fs.writeFileSync(path.join(output,'browser_verification.json'),JSON.stringify({status:'passed',checks,errors,driver_version:require(path.join(process.env.GRIDCAST_PLAYWRIGHT || '/tmp/gridcast-dashboard-browser/node_modules/playwright','package.json')).version},null,2));
    console.log(JSON.stringify({status:'passed',output,checks},null,2));
  } catch (error) {
    await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});
    fs.writeFileSync(path.join(output,'failure.txt'),String(error));
    console.error(String(error));process.exitCode=1;
  } finally {await browser.close();}
})();

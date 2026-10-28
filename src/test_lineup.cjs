const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const sandbox={window:{},document:{getElementById:()=>null},fetch:()=>new Promise(()=>{}),console};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist/app.js'),'utf8'),sandbox);
const solve=sandbox.window.FantasyScout.optimizeLineup;
function brute(pool){const slots=[['QB'],['RB'],['RB'],['WR'],['WR'],['TE'],['RB','WR','TE']];let best=-Infinity;
 function visit(i,used,total){if(i===slots.length){best=Math.max(best,total);return;}for(const p of pool)if(!used.has(p.id)&&slots[i].includes(p.position)){used.add(p.id);visit(i+1,used,total+p.prediction);used.delete(p.id);}}
 visit(0,new Set(),0);return best;
}
let seed=42;const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/2**32;};
for(let trial=0;trial<20;trial++){
 const positions=['QB','QB','RB','RB','RB','WR','WR','WR','TE','TE'];
 const pool=positions.map((position,i)=>({id:String(i),position,prediction:Math.round(rand()*300)/10}));
 const got=solve(pool);assert.ok(got);assert.equal(new Set(got.selected.map(s=>s.p.id)).size,7);
 assert.ok(Math.abs(got.total-brute(pool))<1e-8,'Greedy-per-flex must match exhaustive assignment.');
}
assert.equal(solve([{id:'1',position:'QB',prediction:20}]),null);
console.log('PASS: 20 randomized lineup pools match exhaustive optimum; unique players; incomplete roster rejected.');

const isUpcoming=sandbox.window.FantasyScout.isUpcoming;
assert.equal(isUpcoming({kickoff_utc:'2026-10-30T00:15:00Z'},Date.parse('2026-10-27T05:00:00Z')),true);
assert.equal(isUpcoming({kickoff_utc:'2026-10-30T00:15:00Z'},Date.parse('2026-10-30T00:15:00Z')),false);
assert.equal(isUpcoming({kickoff_utc:'2026-10-30T00:15:00Z'},Date.parse('2026-10-31T05:00:00Z')),false);
console.log('PASS: kickoff-time guard hides started and expired forecasts.');
const available=sandbox.window.FantasyScout.availabilityOK;
const checked={lineup_eligible:true,availability_as_of_utc:'2026-10-27T14:00:00Z'};
assert.equal(available(checked,Date.parse('2026-10-28T14:00:00Z')),true);
assert.equal(available(checked,Date.parse('2026-10-30T14:00:00Z')),false);
assert.equal(available({...checked,lineup_eligible:false},Date.parse('2026-10-28T14:00:00Z')),false);
assert.equal(available({},Date.parse('2026-10-28T14:00:00Z')),false);
console.log('PASS: stale, missing, or excluded availability cannot qualify for automatic lineup.');

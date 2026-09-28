/* applied before first paint so a saved light theme does not flash dark */
try{if(localStorage.getItem("waferscan-theme")==="light")document.documentElement.dataset.theme="light"}catch(e){}

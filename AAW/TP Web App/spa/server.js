const express = require('express');
const req = require('express/lib/request');
const res = require('express/lib/response');
const app = express();
const port = 3000;

const apiRouter = require("./src/app-router");
app.use("/api",apiRouter);

app.use('/%PAH%',express.static('dist'));
app.use('/',express.static('public'));

app.get('/',(req,res) => {
    res.send('Hello World!')
})

app.listen(port, () => {
    console.log(`Example app listening in port ${port}`)
})


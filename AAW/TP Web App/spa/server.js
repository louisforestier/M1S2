const express = require('express');
const app = express();
const port = 3000;

const apiRouter = require("./src/app-router");
app.use(express.json());
app.use("/api", apiRouter);
app.use('/', express.static('dist'));
app.use('/', express.static('public'));

app.get('/', (req, res) => {
    res.send('Hello World!')
})

app.listen(port, () => {
    console.log(`Example app listening in port ${port}`)
})



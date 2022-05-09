const express = require("express");
const router = express.Router();

let persons = [
    {firstName:'Louis',lastName:'Forestier'},
    {firstName:'Clémentine',lastName:'Guillot'}]

router.get("/",(req,res) => {
    res.send("TODO");

})
router.post("/",(req,res) => {
    res.send("TODO");

})
router.delete("/:id",(req,res) => {
    res.send("TODO");

})

module.exports= {
    personRouter:router,
    persons
}
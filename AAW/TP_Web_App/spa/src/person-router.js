const express = require("express");
const router = express.Router();

let persons = [
    {firstName:'Louis',lastName:'Forestier'},
    {firstName:'Clémentine',lastName:'Guillot'}]

router.get("/",(req,res) => {
    res.send(persons);

})
router.post("/",(req,res) => {
    res.send(persons);

})
router.delete("/:id",(req,res) => {
    res.send(persons);

})

module.exports= {
    personRouter:router,
    persons
}
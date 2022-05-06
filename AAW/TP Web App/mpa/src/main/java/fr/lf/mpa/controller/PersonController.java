package fr.lf.mpa.controller;

import fr.lf.mpa.form.PersonForm;
import fr.lf.mpa.service.PersonService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@Controller
public class PersonController extends BaseController {

    @Autowired
    private PersonService personService;

    @GetMapping(value = {"/persons"})
    public String showPersonListPage(Model model) {
        initModel(model);
        if (personService.withContext())
            model.addAttribute("persons", personService.getContext().getPersons());
        else model.addAttribute("persons", personService.getPersons());
        return "person_list";
    }

    @GetMapping(value = {"/persons/add"})
    public String showAddPersonPage(Model model, @RequestParam(required = false) UUID event) {
        initModel(model);
        PersonForm personForm = new PersonForm();
        if (personService.withContext())
            model.addAttribute("events", personService.getContext().getEvents());
        else model.addAttribute("events", personService.getEvents());
        model.addAttribute("personForm", personForm);
        return "add_person";
    }

    @PostMapping(value = {"/persons"})
    public String savePerson(Model model, @ModelAttribute("personForm") PersonForm personForm) {
        initModel(model);
        personService.addPerson(personForm);
        return "redirect:/persons";
    }

    @DeleteMapping(value = {"/persons/{id}"})
    public String delPerson(Model model, @PathVariable("id") UUID id) {
        if (id != null) {
            personService.delPerson(id);
            return "redirect:/persons";
        }
        return "index";
    }
}

package fr.lf.mpa.model;

import fr.lf.mpa.form.PersonForm;

import java.util.UUID;

public class PersonRecord {

    private UUID id;
    private String firstName;
    private String lastName;
    private EventRecord event;

    public PersonRecord(UUID id, String firstName, String lastName, EventRecord event) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.event = event;
    }

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public EventRecord getEvent() {
        return event;
    }

    public void setEvent(EventRecord event) {
        this.event = event;
    }
}

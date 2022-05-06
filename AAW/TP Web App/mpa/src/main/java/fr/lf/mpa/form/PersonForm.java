package fr.lf.mpa.form;

import fr.lf.mpa.model.EventRecord;

import java.util.UUID;

public class PersonForm {
    private String firstName;
    private String lastName;
    private UUID eventId;

    public PersonForm() {

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

    public UUID getEventId() {
        return eventId;
    }

    public void setEventId(UUID eventId) {
        this.eventId = eventId;
    }
}

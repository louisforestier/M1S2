package fr.lf.mpa.model;

import java.util.UUID;

public class EventRecord {

    private UUID id;
    private String name;
    private String date;

    public EventRecord(UUID id, String name, String date) {
        this.id = id;
        this.name = name;
        this.date = date;
    }

    public EventRecord() {
        id=UUID.randomUUID();
    }

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }
}

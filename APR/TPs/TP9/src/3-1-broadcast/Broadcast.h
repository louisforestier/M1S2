#pragma once

/* Effectuer une diffusion en pipeline ...  */
void Broadcast(
    const int k, // num�ro du processeur �metteur
    int *const addr, // entiers � envoyer
    const int N // nombre d'entiers � envoyer au total
);

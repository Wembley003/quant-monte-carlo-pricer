-- Esquema para Microsoft Access (.accdb): tablas runs y greeks.
--
-- Esto es SOLO el esquema (DDL), no una integración con Access. No hay
-- base .accdb aquí y no se ejecutó en Access: en Linux no hay Access.
-- Para usarlo: en Access, Crear > Diseño de consulta > vista SQL,
-- pegar este script y ejecutarlo; las relaciones entre runs y greeks
-- se crean después en Herramientas de base de datos > Relaciones.
-- Tipos: AUTOINCREMENT es el contador de Access, DOUBLE es real de
-- doble precisión y DATETIME es fecha+hora.

CREATE TABLE runs (
    id              AUTOINCREMENT PRIMARY KEY,
    created_at      DATETIME NOT NULL,
    s0              DOUBLE NOT NULL,
    k               DOUBLE NOT NULL,
    t               DOUBLE NOT NULL,
    r               DOUBLE NOT NULL,
    sigma           DOUBLE NOT NULL,
    n_paths         LONG NOT NULL,
    option          TEXT(10) NOT NULL,
    seed            LONG,
    mc_price        DOUBLE NOT NULL,
    mc_se           DOUBLE NOT NULL,
    bs_price        DOUBLE NOT NULL
);

CREATE TABLE greeks (
    id              AUTOINCREMENT PRIMARY KEY,
    run_id          LONG NOT NULL,
    name            TEXT(20) NOT NULL,
    mc_value        DOUBLE NOT NULL,
    bs_value        DOUBLE NOT NULL
);

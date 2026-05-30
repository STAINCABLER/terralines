Container Health Check Action
===========================

Beschreibung
-----------
Diese Composite-Action startet ein lokal verfügbares Container-Image (z. B. direkt nach einem Build mit `load: true`) und prüft periodisch die HTTP-Endpunkte `/health` (mit Fallback `/`) auf Port `8000`. Falls der Container nicht rechtzeitig reagiert, schlägt die Action fehl und schreibt die Container-Logs in `health_container.log`.

Inputs
------
- `image` (required): Image-Referenz, z.B. `ghcr.io/OWNER/IMAGE:tag`.
- `container_name` (optional): Name für den temporären Container. Standard: `container-health-<run_id>`.
- `port` (optional): Port, auf dem der Service horcht (Default: `8000`).
- `retries` (optional): Anzahl der Versuche (Default: `30`).
- `interval` (optional): Sekunden zwischen den Versuchen (Default: `2`).

Beispiel (in einem Workflow)
----------------------------
    - name: Container health check
      uses: ./.github/actions/container-health-check
      with:
        image: ghcr.io/${{ github.repository_owner }}/ltm_ptb_terralines:nightly-${{ github.run_id }}
        container_name: terralines-nightly-${{ github.run_id }}
        port: '8000'
        retries: '30'
        interval: '2'

Artefakte
---------
Die Action schreibt die Container-Logs nach `health_container.log`. Wenn gewünscht, kann der aufrufende Workflow diese Datei anschließend per `actions/upload-artifact` hochladen:

    - name: Upload health check logs
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: health-check-log
        path: health_container.log

Hinweise
------
- Die Action setzt voraus, dass der Runner Docker ausführen kann (Standard `ubuntu-latest` hat Docker verfügbar).
- Die Action ist darauf ausgelegt, Images lokal auszuführen (z.B. nach `docker/build-push-action` mit `load: true`).

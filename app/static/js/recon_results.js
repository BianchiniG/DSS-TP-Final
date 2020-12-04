$(document).ready(function() {
    setTimeout(obtener_resultados, 1000);
})

function obtener_resultados() {
    const rf_class = '.rf';
    const svm_class = '.svm';
    const cnn_class = '.cnn';
    let rf_data = null;
    let svm_data = null;
    let cnn_data = null;

    $.ajax({
        url: '/recon_results',
        success: function(respuesta) {
            if (respuesta !== undefined) {
                if (respuesta.hasOwnProperty(rf_class) && Object.keys(respuesta.rf).length === 0) {
                    // Parsear la respuesta en la variable rf_data
                }
                if (respuesta.hasOwnProperty(svm_class) && Object.keys(respuesta.svm).length === 0) {
                    // Escribir la respuesta en la variable svm_data
                }
                if (respuesta.hasOwnProperty(cnn_class) && Object.keys(respuesta.cnn).length === 0) {
                    // Escribir la respuesta en la variable cnn_data
                }
            }

            // Acá habría que ver si no puede ser una sola funcion generica onda:
            // escribir_respuesta(<clase>, <datos>);
            escribir_respuesta_rf(rf_data);
            escribir_respuesta_svm(svm_data);
            escribir_respuesta_cnn(cnn_data);
        },
        error: function() {
            $(".error").innerHTML("No se ha podido obtener la información");
            // Acá habría que ver si no puede ser una sola funcion generica onda:
            // escribir_respuesta(<clase>, <datos>);
            escribir_respuesta_rf(rf_data);
            escribir_respuesta_svm(svm_data);
            escribir_respuesta_cnn(cnn_data);
        }
    });
}

function escribir_respuesta_rf(data) {
    let html = 'Sin Datos';

    if (data !== null) {
        html = 'Los datos';
    }

    $(rf_class).html(html);
}

function escribir_respuesta_svm(data) {
    let html = 'Sin Datos';

    if (data !== null) {
        html = 'Los datos';
    }

    $(svm_class).html(html);
}

function escribir_respuesta_cnn(data) {
    let html = 'Sin Datos';

    if (data !== null) {
        html = 'Los datos';
    }

    $(cnn_class).html(html);
}
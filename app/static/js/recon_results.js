$(document).ready(function() {
    window.rf_class = '.rf';
    window.svm_class = '.svm';
    window.cnn_class = '.cnn';
    window.rf_data = null;
    window.svm_data = null;
    window.cnn_data = null;

    window.setInterval(function() {
        // obtener_resultados();
    }, 1000);
})

function obtener_resultados() {
    $.ajax({
        url: '/recon_results',
        success: function(respuesta) {
            escribir_respuestas(respuesta);
        },
        error: function() {
            $(".error").html("No se ha podido obtener la información");
        }
    });
}

function escribir_respuestas(respuesta) {
    if (respuesta !== undefined && respuesta !== null) {
        if (respuesta.rf) {
            window.rf_data = respuesta.rf;
        }
        if (respuesta.svm) {
            window.svm_data = respuesta.svm;
        }
        if (respuesta.cnn) {
            window.cnn_data = respuesta.cnn;
        }
    }

    $(window.rf_class).html(window.rf_data ? window.rf_data : "Sin datos");
    $(window.svm_class).html(window.svm_data ? window.svm_data : "Sin datos");
    $(window.cnn_class).html(window.cnn_data ? window.cnn_data : "Sin datos");
}

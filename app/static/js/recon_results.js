$(document).ready(function() {
    window.setInterval(function() {
        obtener_resultados();
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
    if (respuesta.imagen != null && respuesta.predicciones != null) {
        const detecciones_content = $('.detecciones-realtime');
        borrar_elementos_viejos(detecciones_content);

        detecciones_content.prepend('' +
            '<div class="row" style="padding: 10px 0 10px 0;">' +
            '  <div class="col-md-4">' +
            '    <img src="'+respuesta.imagen+'" width="100%">' +
            '  </div>' +
            '  <div class="col-md-7" style="text-align: left;">' +
            '    <p>Random Forest: '+respuesta.predicciones.rf+'<br>' +
            '    SVM: '+respuesta.predicciones.svm+'<br>' +
            '    Red Neuronal: '+respuesta.predicciones.cnn+'</p>' +
            '  </div>' +
            '</div>');
    }
}

function borrar_elementos_viejos(detecciones_content) {
    let elementos = detecciones_content.children('div');
    let elementos_a_borrar = elementos.slice(2, elementos.length)
    for (let i = 0; i < elementos_a_borrar.length; i++) {
        elementos_a_borrar[i].remove();
    }
}

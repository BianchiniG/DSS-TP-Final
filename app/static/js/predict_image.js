$(document).ready(function() {
    window.rf_class = '.rf';
    window.svm_class = '.svm';
    window.cnn_class = '.cnn';
    window.rf_data = null;
    window.svm_data = null;
    window.cnn_data = null;

    $("form").submit(function(e) {
        e.preventDefault(e);
    });

    $("#procesar-imagen").on('click', function() {
        $('#procesar-imagen').attr('disabled', true);
        $('#procesar-imagen').html("Procesando...");
        const input = $('#foto');
        if (input.files && input.files[0]) {
            $('#error-load').html("");
            let reader = new FileReader();

            reader.onload = function (e) {
                const base64_image = e.target.result;
                $.ajax({
                    url: '/process_image',
                    data: base64_image,
                    success: function(respuesta) {
                        escribir_respuestas(respuesta);
                    },
                    error: function() {
                        $(".error").html("No se ha podido obtener la información");
                    },
                    finally: function() {
                        $('#procesar-imagen').html("¡Procesar imagen!");
                        $('#procesar-imagen').attr('disabled', false);
                    }
                });
            };

            reader.readAsDataURL(input.files[0]);
        } else {
            $('#error-load').html("Debe seleccionar una imágen!");
        }



    });
});

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

function load_image(input) {
    if (input.files && input.files[0]) {
        $('#selected-image').css('display', 'initial');
        let reader = new FileReader();

        reader.onload = function (e) {
            $('#selected-image')
                .attr('src', e.target.result)
                .width('10%');
        };

        reader.readAsDataURL(input.files[0]);
    } else {
        $('#selected-image').css('display', 'none');
    }
}

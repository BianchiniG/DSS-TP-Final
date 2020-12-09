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
    const image_proc_btn = $('#procesar-imagen');

    $("#procesar-imagen").on('click', function() {
        image_proc_btn.attr('disabled', true);
        image_proc_btn.html("Procesando...");

        var form_data = new FormData();
        const input = $('#foto')[0].files;
        
        if (input.length>0) {
            form_data.append('file',input[0])
            $.ajax({
                    type:'POST',
                    url: 'http://localhost:5001/process_image',
                    data: form_data,
                    contentType: false,
                    processData: false,
                    success: function(respuesta) {
                        escribir_respuestas(respuesta);
                        image_proc_btn.html("¡Procesar imagen!");
                        image_proc_btn.attr('disabled', false);
                    },
                    error: function() {
                        $(".error-load").html("No se ha podido obtener la información");
                        image_proc_btn.html("¡Procesar imagen!");
                        image_proc_btn.attr('disabled', false);
                    },
                    finally: function() {
                        image_proc_btn.html("¡Procesar imagen!");
                        image_proc_btn.attr('disabled', false);
                    }
                });
        } else {
            $('.error-load').html("Debe seleccionar una imágen!");
        }
    });
});

function escribir_respuestas(respuesta) {
    if (respuesta !== undefined && respuesta !== null) {
        console.log(respuesta)
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
